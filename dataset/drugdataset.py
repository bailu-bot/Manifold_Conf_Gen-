import os
import os.path as osp
import json
import pickle


import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from rdkit import Chem
from tqdm import tqdm

from .smiles2graph import smile2graph4GEOM



class QM9Dataset(InMemoryDataset):
    def __init__(self, name, root='data',dataset='QM9',
                 transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.root = root
        # self.dir_name = '_'.join(name.split('-'))
        self.type = type
        super(QM9Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_index, self.valid_index, self.test_index = pickle.load(open(self.processed_paths[1], 'rb'))
        self.num_tasks = 1

    @property
    def raw_dir(self):

        return 'GEOM_Data'
    

    @property
    def raw_file_names(self):
      
        return 'GEOM_Data/QM9/train_converted_data5K.pt', 'GEOM_Data/QM9/val_converted_data2K.pt', 'GEOM_Data/QM9/test_converted_data2K.pt'


    @property
    def processed_dir(self):

        return 'processed_data'

    @property
    def processed_file_names(self):
        return 'data.pt','split.pt'

    def __subprocess(self, datalist):
        processed_data = []
        for datapoint in tqdm(datalist):
            smiles = datapoint['smiles']
            mol = Chem.MolFromSmiles(smiles)

            if (mol is not None):
                mol = Chem.AddHs(mol)
                if(mol.GetNumAtoms()==datapoint['pos'].shape[0]):              
                    x, edge_index, edge_attr = smile2graph4GEOM(smiles)
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles,pos=datapoint['pos'],
                                boltzmannweight=datapoint['boltzmannweight'], idx=datapoint['idx'],rdmol=datapoint['rdmol'],
                                totalenergy=datapoint['totalenergy'], atom_type=datapoint['atom_type'])
                    
                    print(data.x.shape,data.pos.shape,data.atom_type)
                    data.batch_num_nodes = data.num_nodes
                # if self.pre_filter is not None and not self.pre_filter(data):
                #     continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    processed_data.append(data)
            else :
                continue
        return processed_data, len(processed_data)

    def process(self):
        # data_list = []
        train_data = torch.load('GEOM_Data/QM9/train_converted_data5K.pt')
        valid_data = torch.load('GEOM_Data/QM9/val_converted_data2K.pt')
        test_data  = torch.load('GEOM_Data/QM9/test_converted_data2K.pt')
        train_data_list, train_num = self.__subprocess(train_data)
        valid_data_list, valid_num = self.__subprocess(valid_data)
        test_data_list, test_num = self.__subprocess(test_data)
        data_list = train_data_list + valid_data_list + test_data_list
        train_index = list(range(train_num))
        valid_index = list(range(train_num, train_num + valid_num))
        test_index = list(range(train_num + valid_num, train_num + valid_num + test_num))
        torch.save(self.collate(data_list), self.processed_paths[0])
        pickle.dump([train_index, valid_index, test_index], open(self.processed_paths[1], 'wb'))


    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
