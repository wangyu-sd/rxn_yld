# !/usr/bin/python3
# @File: dataset.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.18.21
import os
import os.path as osp

import pandas as pd
import torch
from rdkit import RDLogger
from torch_geometric.data import Dataset
from tqdm import tqdm

from data.data_utils import smile_to_mol_info, get_tgt_adj_order, \
    get_bond_order_adj, rxn_dict_process

RDLogger.DisableLog('rdApp.*')


class RXNDataSet(Dataset):
    @property
    def raw_file_names(self):
        return [self.data_split + ".csv"]

    @property
    def processed_file_names(self):
        return [f"rxn_data_{idx}.pt" for idx in range(self.size)]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, osp.join("processed", self.data_split))

    def __init__(self, root, data_split, fast_read=True, max_node=200, min_node=5, save_cache=True, use_3d_info=False):
        self.root = root
        # self.rxn_center_path = osp.join(root, osp.join('processed', 'rxn_center.pt'))
        self.min_node = min_node
        self.max_node = max_node
        self.data_split = data_split
        self.use_3d_info = use_3d_info

        self.size_path = osp.join(osp.join(self.root, osp.join("processed", self.data_split)), "num_files.pt")
        if osp.exists(self.size_path):
            self.size = torch.load(self.size_path)
        else:
            self.size = 0
        super().__init__(root)
        if fast_read:
            data_cache_path = osp.join(self.root, osp.join('processed', f'cache_{data_split}.pt'))
            if osp.isfile(data_cache_path) and save_cache:
                print(f"read cache from {data_cache_path}...")
                self.data = torch.load(data_cache_path)
            else:
                self.fast_read = False
                self.data = [rxn_data for rxn_data in tqdm(self)]
                if save_cache:
                    torch.save(self.data, data_cache_path)

        self.fast_read = fast_read

        # print(self.data[0]['mask'].size())
        # for mol in ["product", "reactant"]:
        #     for k, v in self.data[0][mol].items():
        #         if isinstance(v, torch.Tensor):
        #             print(mol, k, v.size())
        #         else:
        #             print(mol, k, v)

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        for raw_file_name in self.raw_file_names:
            print(f"Processing the {raw_file_name} dataset to torch geometric format...\n")

            csv = pd.read_csv(osp.join(self.raw_dir, raw_file_name))
            prods = csv['product']
            rcts = csv['reactant']
            clts = csv['catalyst']
            rxn_ids = csv['Reaction_ID']
            total = len(prods)

            cur_id = 0
            for idx, prod in tqdm(enumerate(prods), total=total):
                # print(rcts.iloc[idx], type(clts.iloc[idx]) == float)
                try:
                    product = smile_to_mol_info(prods.iloc[idx], use_3d_info=self.use_3d_info, obj='product')
                    reactant = smile_to_mol_info(rcts.iloc[idx], use_3d_info=self.use_3d_info, obj='reactant')
                    all_atom = 0
                    if isinstance(clts.iloc[idx], float):
                        catalyst = None
                    else:
                        catalyst = smile_to_mol_info(clts.iloc[idx], use_3d_info=self.use_3d_info, obj='catalyst')
                        all_atom += catalyst['n_atom']
                except AttributeError:
                    continue

                order = get_tgt_adj_order(product['mol'], reactant['mol'])
                reactant['n_atom'] = order.size(0)
                all_atom += max(reactant['n_atom'], product['n_atom'])
                if all_atom > self.max_node or all_atom < self.min_node:
                    continue

                reactant['atom_fea'] = reactant['atom_fea'][:, order]
                reactant['bond_adj'] = reactant['bond_adj'][order, :][:, order]
                reactant['dist_adj'] = reactant['dist_adj'][order, :][:, order]

                n_pro, n_rea = product['n_atom'], reactant['n_atom']

                pro_bond_adj = get_bond_order_adj(product['mol'])
                rea_bond_adj = get_bond_order_adj(reactant['mol'])[order][:, order]
                rc_target = torch.zeros_like(pro_bond_adj)
                rc_target[:n_pro, :n_pro] = rea_bond_adj[:n_pro, :n_pro]

                rc_target = (~torch.eq(rc_target, pro_bond_adj))
                center = rc_target.nonzero()
                center_cnt = center.size(0) // 2
                if center_cnt >= 50:
                    continue

                for i, j in center:
                    if i > j:
                        reactant['bond_adj'][i, j] += (1 << 6)
                        reactant['bond_adj'][j, i] += (1 << 6)
                        product['bond_adj'][i, j] += (1 << 6)
                        product['bond_adj'][j, i] += (1 << 6)
                        product['atom_fea'][9, i] += 1
                        product['atom_fea'][9, j] += 1
                        reactant['atom_fea'][9, i] += 1
                        reactant['atom_fea'][9, j] += 1

                if catalyst is not None:
                    product, reactant = rxn_dict_process(product, reactant, catalyst)
                rxn_data = (
                    float(csv['Y'].iloc[idx]),
                    product,
                    reactant,
                    all_atom,
                    center_cnt,
                    rxn_ids.iloc[idx]
                )
                torch.save(rxn_data, osp.join(self.processed_dir, f"rxn_data_{cur_id}.pt"))
                cur_id += 1

            print(f"Completed the {raw_file_name} dataset to torch geometric format...")

        # cnt = 0
        # print(f"|{self.data_split}: {len(leaving_group)}|", "*" * 89)
        # print("type\tnum\tatoms")
        # for lg in leaving_group:
        #     print(f"{lg.rxn_type}\t{lg.n}\t{lg.atoms}")
        #     cnt += lg.n
        print(f"|total={cur_id}|\t|passed={idx - cur_id}|", '*' * 90)
        self.size = cur_id
        torch.save(self.size, self.size_path)

    def len(self):
        return self.size

    def get(self, idx):
        if self.fast_read:
            rxn_data = self.data[idx]
        else:
            rxn_data = torch.load(osp.join(self.processed_dir, f"rxn_data_{idx}.pt"))
        return rxn_data
