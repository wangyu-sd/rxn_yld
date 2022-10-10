# !/usr/bin/python3
# @File: datamodual.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.19.12
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from data.datasets import RXNDataSet
import os
import os.path as osp
from tdc.single_pred import Yields
import pandas as pd
from tqdm import tqdm
DS_DIC = {'uspto': "USPTO_Yields", 'buch-hart': "Buchwald-Hartwig"}


class RXNDataModule(pl.LightningDataModule):
    def __init__(
            self,
            root,
            batch_size,
            fast_read=False,
            split_names=None,
            num_workers=None,
            pin_memory=True,
            shuffle=True,
            predict=True,
            seed=114,
            dataset_format='uspto'
    ):
        super().__init__()
        read_raw = False
        exist_process = os.path.exists(os.path.join(root, 'processed/test'))
        if exist_process:
            print(f'dataset in {root} has already been processed, reading from processed directory...')
            if split_names is None and not predict:
                split_names = ["train", "valid"]
            elif split_names is None:
                split_names = ['test']
        else:
            print(f'processing raw file from {root}...')
            split_names = ['train', 'valid', 'test']
            read_raw = not os.path.exists(os.path.join(root, 'raw/test.csv'))

        self.split_names = split_names
        if num_workers is None:
            num_workers = len(os.sched_getaffinity(0))

        if read_raw:
            raw_path = osp.join(root, "raw")
            raw_data = Yields(name=DS_DIC[dataset_format], path=raw_path).get_split(seed=seed)
            for data_split in split_names:
                length = len(raw_data[data_split])
                print(f'Splitting/{data_split}...')
                with open(osp.join(raw_path, data_split + ".csv"), 'w') as f:
                    f.write('Reaction_ID,product,reactant,catalyst,Y\n')
                    for idx in tqdm(range(length)):
                        f.write(raw_data[data_split]['Reaction_ID'].iloc[idx] + ",")
                        for key in ['product', 'reactant', 'catalyst']:
                            f.write(raw_data[data_split].iloc[idx]['Reaction'][key] + ",")
                        f.write(str(raw_data[data_split].iloc[idx]['Y']) + "\n")

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.dataset_dict = {}

        if dataset_format == "uspto":
            max_node = 200
            for data_split in split_names:
                self.dataset_dict[data_split] = RXNDataSet(root=root,
                                                           data_split=data_split,
                                                           fast_read=fast_read,
                                                           max_node=max_node, )
        elif dataset_format == "buch-hart":
            max_node = 200
            atom_mapped(raw_data)
            for data_split in split_names:
                self.dataset_dict[data_split] = RXNDataSet(root=root,
                                                           data_split=data_split,
                                                           fast_read=fast_read,
                                                           max_node=max_node, )
        else:
            raise NotImplementedError

        self.collate_fn = RxnCollector(max_node)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_dict[self.split_names[0]],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_dict[self.split_names[1]],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_dict[self.split_names[0]],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def predict_dataloader(self):
        return self.test_dataloader()


class RxnCollector:
    def __init__(self, max_node):
        self.max_node = max_node

    def __call__(self, data_list):
        batch_size = len(data_list)
        max_atoms = max([data[3] for data in data_list])

        y_list = []
        center_cnt = []
        p_dict = {
            'atom_fea': torch.zeros(batch_size, 11, max_atoms, dtype=torch.half),
            'bond_adj': torch.zeros(batch_size, max_atoms, max_atoms, dtype=torch.uint8),
            'dist_adj': torch.zeros(batch_size, max_atoms, max_atoms, dtype=torch.float),
            'n_atom': []
        }
        r_dict = {
            'atom_fea': torch.zeros(batch_size, 11, max_atoms, dtype=torch.half),
            'bond_adj': torch.zeros(batch_size, max_atoms, max_atoms, dtype=torch.uint8),
            'dist_adj': torch.zeros(batch_size, max_atoms, max_atoms, dtype=torch.float),
            'n_atom': []
        }

        for idx, data in enumerate(data_list):
            y_list.append(data[0])
            n_pro, n_rct = data[1]['n_atom'], data[2]['n_atom']
            p_dict['n_atom'].append(n_pro)
            r_dict['n_atom'].append(n_rct)
            center_cnt.append(data[4])

            p_dict['atom_fea'][idx, :, :n_pro] = data[1]['atom_fea']
            p_dict['bond_adj'][idx, :n_pro, :n_pro] = data[1]['bond_adj'] + 1
            p_dict['dist_adj'][idx, :n_pro, :n_pro] = data[1]['dist_adj']

            # print(data[2]['atom_fea'].size(), n_rct, data[5], data[2]['atom_fea'][0])

            r_dict['atom_fea'][idx, :, :n_rct] = data[2]['atom_fea']
            r_dict['bond_adj'][idx, :n_rct, :n_rct] = data[2]['bond_adj'] + 1
            r_dict['dist_adj'][idx, :n_rct, :n_rct] = data[2]['dist_adj']

        p_dict['n_atom'] = torch.tensor(p_dict['n_atom'])
        r_dict['n_atom'] = torch.tensor(r_dict['n_atom'])
        y_list = torch.tensor(y_list)
        center_cnt = torch.tensor(center_cnt)

        return p_dict, r_dict, center_cnt, y_list


def atom_mapped(raw_data_set):
    pass
