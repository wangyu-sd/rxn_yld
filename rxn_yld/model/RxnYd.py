# !/usr/bin/python3
# @File: RetroAGT.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.20.22
import sys

sys.path.append("..")
sys.path.append("./model")
import os
from typing import List
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import pandas as pd
from torch import nn, Tensor
from model.LearningRate import PolynomialDecayLR
from model.Embeddings import RxnYdEmbeddingLayer
from model.Modules import RxnYdEncoderLayer, RxnYdEncoder, MultiHeadAtomAdj
import copy


class RxnYd(pl.LightningModule):
    def __init__(self, d_model=512, nhead=32, p_layer=2, r_layer=2, output_layer=2, dim_feedforward=512, dropout=0.1,
                 max_single_hop=4, known_rxn_cnt=True, n_layers=1, batch_first=True, norm_first=False,
                 activation='gelu', use_contrastive=False, warmup_updates=6e4, tot_updates=1e6, peak_lr=2e-4,
                 end_lr=1e-9, weight_decay=0.99, use_3d_info=False, use_dist_adj=True):
        super().__init__()
        self.p_layer = p_layer
        self.r_layer = r_layer
        self.max_single_hop = max_single_hop
        self.output_layer = output_layer
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.activation = activation
        self.warmup_updates = warmup_updates
        self.peak_lr = peak_lr
        self.tot_updates = tot_updates
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.use_3d_info = use_3d_info
        self.use_dist_adj = use_dist_adj
        self.known_rxn_cnt = known_rxn_cnt
        self.use_contrastive = use_contrastive

        encoder_layer = RxnYdEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                          batch_first=batch_first, norm_first=norm_first)
        self.p_emb = RxnYdEmbeddingLayer(d_model, nhead, max_single_hop, n_layers, need_graph_token=True,
                                         use_3d_info=use_3d_info, dropout=dropout, known_rxn_cnt=known_rxn_cnt,
                                         use_dist_adj=use_dist_adj)
        self.r_emb = copy.deepcopy(self.p_emb)

        self.p_encoder = RxnYdEncoder(encoder_layer, p_layer, nn.LayerNorm(d_model))
        self.r_encoder = RxnYdEncoder(encoder_layer, r_layer, nn.LayerNorm(d_model))
        self.out_encoder = RxnYdEncoder(encoder_layer, output_layer, nn.LayerNorm(d_model))

        self.out_fn = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, p_dict, r_dict, center_cnt):
        p_atom, p_adj = self.p_emb(p_dict['atom_fea'], p_dict['bond_adj'], p_dict['dist_adj'], center_cnt)
        p_atom_fea = self.p_encoder(p_atom, p_adj)

        r_atom, r_adj = self.r_emb(r_dict['atom_fea'], r_dict['bond_adj'], r_dict['dist_adj'], center_cnt)
        r_atom_fea = self.p_encoder(r_atom, r_adj)

        out_atom_fea = (p_atom_fea + r_atom_fea) / 2
        out_adj = (p_adj + r_adj) / 2
        out_atom_fea = self.out_encoder(out_atom_fea, out_adj)[:, 0]
        y_pred = self.out_fn(out_atom_fea)

        return y_pred

    def training_step(self, batch, batch_idx):
        p_dict, r_dict, center_cnt, y = batch
        y_pred = self.forward(p_dict, r_dict, center_cnt)
        loss_mse = self.criterion(y, y_pred)
        mae = F.l1_loss(y, y_pred)
        self.log("mse", loss_mse, prog_bar=True, logger=True)
        self.log("mae", mae, prog_bar=True, logger=True)

        return loss_mse

    def validation_step(self, batch, batch_idx):
        p_dict, r_dict, center_cnt, y = batch
        y_pred = self.forward(p_dict, r_dict, center_cnt)
        loss_mse = self.criterion(y, y_pred)
        mae = F.l1_loss(y, y_pred)
        self.log("mse", loss_mse, prog_bar=True, logger=True)
        self.log("mae", mae, prog_bar=True, logger=True)
        return {
            'mse': loss_mse,
            'mae': mae,
        }

    def validation_epoch_end(self, outputs):
        self._log_dict(self._avg_dicts(outputs))

    def test_step(self, batch, batch_idx):
        p_dict, r_dict, center_cnt, y = batch
        y_pred = self.forward(p_dict, r_dict, center_cnt)
        loss_mse = self.criterion(y, y_pred)
        mae = F.l1_loss(y, y_pred)
        self.log("mse", loss_mse, prog_bar=True, logger=True)
        self.log("mae", mae, prog_bar=True, logger=True)
        return {
            'mse': loss_mse,
            'mae': mae,
        }

    def test_epoch_end(self, outputs):
        self._log_dict(self._avg_dicts(outputs))

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        p_dict, r_dict, center_cnt, y = batch
        y_pred = self.forward(p_dict, r_dict, center_cnt)
        loss_mse = self.criterion(y, y_pred)
        mae = F.l1_loss(y, y_pred)
        self.log("mse", loss_mse, prog_bar=True, logger=True)
        self.log("mae", mae, prog_bar=True, logger=True)
        return loss_mse, mae

    @staticmethod
    def _avg_dicts(colls):
        complete_dict = {key: [] for key, val in colls[0].items()}
        for coll in colls:
            [complete_dict[key].append(coll[key]) for key in complete_dict.keys()]
        avg_dict = {key: sum(l) / len(l) for key, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val, sync_dist=True)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--seed', type=int, default=123)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--nhead', type=int, default=16)
        parser.add_argument('--p_layer', type=int, default=2)
        parser.add_argument('--r_layer', type=int, default=2)
        parser.add_argument('--output_layer', type=int, default=2)
        parser.add_argument('--n_rxn_type', type=int, default=10)
        parser.add_argument('--dim_feedforward', type=int, default=512)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--batch_second', default=False, action='store_true')
        parser.add_argument('--known_rxn_cnt', default=False, action='store_true')
        parser.add_argument('--norm_first', default=False, action='store_true')
        parser.add_argument('--activation', type=str, default='gelu')
        parser.add_argument("--warmup_updates", type=int, default=6000)
        parser.add_argument("--tot_updates", type=int, default=200000)
        parser.add_argument("--peak_lr", type=float, default=2e-4)
        parser.add_argument("--end_lr", type=float, default=1e-9)
        parser.add_argument('--weight_decay', type=float, default=1e-2)
        parser.add_argument('--max_single_hop', type=int, default=4)
        parser.add_argument('--not_use_dist_adj', default=False, action='store_true')
        parser.add_argument('--not_use_contrastive', default=False, action='store_true')
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay
        )
        lr_scheduler = {
            "scheduler": PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
