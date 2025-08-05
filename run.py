import os
import logging
from tqdm import tqdm
from munch import Munch, munchify
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from GOOD import register
from GOOD.utils.config_reader import load_config
from GOOD.utils.metric import Metric
from GOOD.data.dataset_manager import read_meta_info
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.utils.train import nan2zero_get_mask
import matplotlib.pyplot as plt
from args_parse import args_parser
from exputils import initialize_exp, set_seed, get_dump_path, describe_model ,save_checkpoint,load_checkpoint,visualize_mol
from models import GNNEncoder    
from dataset import QM9Dataset
from sklearn.manifold import TSNE
logger = logging.getLogger()
import pickle
import copy

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class EarlyStopping:
    def __init__(self, patience=10, lower_better=True):
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf') if lower_better else -float('inf')
        self.best_model = None
        self.early_stop = False
        self.lower_better = lower_better

    def step(self, score, model):
        improved = score < self.best_score if self.lower_better else score > self.best_score
        if improved:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Runner:
    def __init__(self, args, writer, logger_path):
        self.args = args
        self.device = torch.device(f'cuda')
        self.all_c_features = []
        self.y_labels = []
        dataset = QM9Dataset(name=args.dataset, root=args.data_root)
        self.train_set = dataset[dataset.train_index]
        self.valid_set = dataset[dataset.valid_index]
        self.test_set = dataset[dataset.test_index]
        self.lower_better = 1
        self.train_loader = DataLoader(self.train_set, batch_size=args.bs, shuffle=True, drop_last=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=args.bs, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=args.bs, shuffle=False)
        self.metric = Metric()
        cfg = Munch()
        cfg.metric = self.metric
        cfg.model = Munch() 
        cfg.model.model_level = 'graph'
        self.early_stopper = EarlyStopping(patience=args.patience, lower_better=True)
        self.model = GNNEncoder(args=args, config=cfg).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = CosineAnnealingLR(self.opt, T_max=args.epoch, eta_min=args.eta_min)
        self.total_step = 0
        self.writer = writer
        describe_model(self.model, path=logger_path)
        self.logger_path = logger_path
        self.cfg = cfg

    def run(self):
        if self.lower_better == 1:
            best_valid_score, best_test_score = float('inf'), float('inf')
        else:
            best_valid_score, best_test_score = -1, -1
        if self.args.checkpoint_path is not None:
            checkpoint = torch.load(self.args.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            best_test_score = checkpoint['best_test_score']
            print("use previous checkpoint", "epoch is",epoch, "best_test_score is", best_test_score)
        for e in range(self.args.epoch):
            self.train_step(e)
            valid_score = self.test_step(self.valid_loader)
            
            logger.info(f"E={e}, Metrics:{self.args.metric},    valid={valid_score:.5f}, test-score={best_test_score:.5f}")
            self.scheduler.step()
            self.early_stopper.step(valid_score, self.model)
            if self.early_stopper.early_stop:
                logger.info("Early stopping triggered.")
                self.model.load_state_dict(self.early_stopper.best_model)  # 恢复最佳模型
                break
            # if valid_score > best_valid_score:
            if valid_score < best_valid_score :
                test_score = self.test_step(self.test_loader)
                best_valid_score = valid_score
                best_test_score = test_score
                logger.info(f"Metrics:{self.args.metric},  UPDATE test-score={best_test_score:.5f}")
                save_checkpoint(self.model, self.opt, e, best_test_score, os.path.join(self.logger_path, 'best.pth'))
            if e % 10 == 0:
                save_checkpoint(self.model, self.opt, e, best_test_score, os.path.join(self.logger_path, f'epoch{e}.pth'))
        logger.info(f"test-score={best_test_score:.5f},learning rate={self.opt.param_groups[0]['lr']}")

    def train_step(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"E [{epoch}]")
        self.all_c_features = []
        for data in pbar:
            data = data.to(self.device)
            pred_pos,graph_feat, pos_loss, mani_loss = self.model(data)
            #print(data.smiles)
            if self.args.get_image==True:
                for i, smiles in enumerate(data.smiles):
                    if smiles==self.args.smiles:
                        mol=data.rdmol[i]
                        node_mask = (data.batch == i)            # bool tensor
                        pos_i = pred_pos[node_mask]               # [num_atoms_i, 3]

                        # 把预测坐标写回到 mol 上
                        conf = Chem.Conformer(mol.GetNumAtoms())
                        for atom_idx, (x, y, z) in enumerate(pos_i.tolist()):
                            conf.SetAtomPosition(atom_idx, Point3D(x, y, z))

                        mol.AddConformer(conf, assignId=True)
            
        if epoch==self.args.epoch-1:
            saved=False
            for data in pbar:
                data = data.to(self.device)
                for i, smiles in enumerate(data.smiles):
                    if smiles==self.args.smiles and saved==False:
                        mol_gt = copy.deepcopy(mol)       
                        mol_gt.RemoveAllConformers()      # 先清空它身上的所有 conformer

                        # 从 data.pos 中取出 ground-truth 的那张图的所有原子位置
                        node_mask = (data.batch == i)
                        pos_gt = data.pos[node_mask]      # tensor [num_atoms,3]

                        # 新建一个 conformer 并填坐标
                        conf_gt = Chem.Conformer(mol_gt.GetNumAtoms())
                        for atom_idx, (x, y, z) in enumerate(pos_gt.tolist()):
                            conf_gt.SetAtomPosition(atom_idx, Point3D(x, y, z))
                        mol_gt.AddConformer(conf_gt, assignId=True)

                        output_dir = f"rdmol/{smiles}"
                        os.makedirs(output_dir, exist_ok=True)   
                        file_path_gt = os.path.join(output_dir, f"mol_view_{smiles}_gt.pkl")
                        file_path = os.path.join(output_dir, f"mol_view_{smiles}.pkl")
                        with open(file_path, "wb") as f:
                            pickle.dump(mol, f)
                        with open(file_path_gt, "wb") as f:
                            pickle.dump(mol_gt, f)
                        print("successfully saved mol view to", output_dir)
                        saved = True
                        break
                if saved==True:
                    break
                        


            loss = pos_loss * self.args.pos_w + mani_loss * self.args.mani_w 
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.opt.step()
            
            pbar.set_postfix_str(f"loss={loss.item():.4f}, mani_loss={mani_loss.item():.4f}")
            self.writer.add_scalar('loss', loss.item(), self.total_step)
            self.writer.add_scalar('pos-loss', pos_loss.item(), self.total_step)
            self.writer.add_scalar('mani-loss', mani_loss.item(), self.total_step)
            self.writer.add_scalar('lr', self.opt.param_groups[0]['lr'], epoch)

            self.total_step += 1
        # if epoch % 10 == 0 or epoch == self.args.epoch-1:
        #     self.visualize_aggregated_tsne(epoch)
        
    @torch.no_grad()
    def test_step(self, loader):
        self.model.eval()
        y_pred, y_gt = [], []

        for data in loader:
            data = data.to(self.device)

            pred_pos, _, _, _ = self.model(data)

            y_pred.append(pred_pos)         # shape: [num_atoms, 3]
            y_gt.append(data.pos)           # ground truth positions

        # 把 list of tensors → tensor
        y_pred = torch.cat(y_pred, dim=0)  # shape: [total_atoms, 3]
        y_gt = torch.cat(y_gt, dim=0)      # same shape
        if self.args.metric == 'MAE':
            score = F.l1_loss(y_pred, y_gt)
        elif self.args.metric == 'RMSD':
            score = torch.sqrt(F.mse_loss(y_pred, y_gt))
        else:
            raise ValueError(f"metric is {self.args.metric}. We need MAE or RMSD.")
        return score

def main():
    args = args_parser()
    torch.cuda.set_device(int(args.gpu))

    logger = initialize_exp(args)
    set_seed(args.random_seed)
    logger_path = get_dump_path(args)
    writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard'))

    runner = Runner(args, writer, logger_path)
    runner.run()
    writer.close()


if __name__ == '__main__':
    main()
