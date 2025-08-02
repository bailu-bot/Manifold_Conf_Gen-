import os
import logging
from tqdm import tqdm
from munch import Munch, munchify

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import numpy as np

from GOOD import register
from GOOD.utils.config_reader import load_config
from GOOD.utils.metric import Metric
from GOOD.data.dataset_manager import read_meta_info
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.utils.train import nan2zero_get_mask
import matplotlib.pyplot as plt
from args_parse import args_parser
from exputils import initialize_exp, set_seed, get_dump_path, describe_model ,save_checkpoint,load_checkpoint
from models import GNNEncoder    
from dataset import QM9Dataset
from sklearn.manifold import TSNE
logger = logging.getLogger()


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
        self.metric.set_loss_func(task_name='Binary classification')
        self.metric.set_score_func(metric_name='ROC-AUC')
        cfg = Munch()
        cfg.metric = self.metric
        cfg.model = Munch()
        cfg.model.model_level = 'graph'

        self.model = GNNEncoder(args=args, config=cfg).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)

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
            
            logger.info(f"E={e}, valid={valid_score:.5f}, test-score={best_test_score:.5f}")
            # if valid_score > best_valid_score:
            if valid_score < best_valid_score :
                test_score = self.test_step(self.test_loader)
                best_valid_score = valid_score
                best_test_score = test_score
                logger.info(f"UPDATE test-score={best_test_score:.5f}")
                save_checkpoint(self.model, self.opt, e, best_test_score, os.path.join(self.logger_path, 'best.pth'))
            if e % 10 == 0:
                save_checkpoint(self.model, self.opt, e, best_test_score, os.path.join(self.logger_path, f'epoch{e}.pth'))
        logger.info(f"test-score={best_test_score:.5f}")

    def train_step(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"E [{epoch}]")
        self.all_c_features = []
        for data in pbar:
            data = data.to(self.device)
            pred_pos,graph_feat, pos_loss, mani_loss = self.model(data)


            loss = pos_loss * self.args.pos_w + mani_loss * self.args.mani_w 
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.opt.step()

            pbar.set_postfix_str(f"loss={loss.item():.4f}","mani_loss={mani_loss.item():.4f}")
            self.writer.add_scalar('loss', loss.item(), self.total_step)
            self.writer.add_scalar('pos-loss', pos_loss.item(), self.total_step)
            self.writer.add_scalar('mani-loss', mani_loss.item(), self.total_step)


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

        score = F.mse_loss(y_pred, y_gt)

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
