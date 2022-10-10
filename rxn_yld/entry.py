# !/usr/bin/python3
# @File: train.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.27.19
import os
import argparse
import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.utilities import seed
from model.RxnYd import RxnYd
from data.datamodule import RXNDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from pytorch_lightning.loggers import TensorBoardLogger

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = RxnYd.add_model_specific_args(parser)

    # trainer configuration
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--acc_batches', type=int, default=8)
    parser.add_argument('--log_dir', type=str, default='tb_logs')
    parser.add_argument('--name', type=str, default='yield')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--predict', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cuda', type=str, default='0')

    # dataset configuration
    parser.add_argument('--dataset', type=str, default="data/uspto")
    parser.add_argument('--not_fast_read', default=False, action='store_true')
    parser.add_argument('--use_3d_info', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset_format', type=str, default='uspto')

    args = parser.parse_args()
    print(args)

    seed.seed_everything(args.seed)

    print("Building DataModule...")
    dm = build_datamodule(args)
    print("Finished DataModule.")

    print("Building Model...")

    model = build_model(args)
    print("Finished Model...")

    print("Building Trainer...")
    trainer = build_trainer(args)
    print("Finished Trainer...")

    if not args.test and not args.predict:
        trainer.fit(model, dm)
        print('Finished training..')
        print(args)

    elif args.test:
        print('Testing...')
        trainer.test(model, dm)
    elif args.predict:
        print('predict...')
        root = os.path.join(args.dataset, 'known_rxn_type' if args.known_rxn_type else 'unknown_rxn_type')
        os.makedirs(root, exist_ok=True)
        outputs = trainer.predict(model, dm)
        complete_dict = {key: [] for key, val in outputs[0].items()}
        for coll in outputs:
            [complete_dict[key].append(coll[key]) for key in complete_dict.keys()]
        for key, val in complete_dict.items():
            complete_dict[key] = torch.cat(val, dim=0)
        torch.save(complete_dict, os.path.join(root, 'tsne_data.pt'))


def build_trainer(args):
    logger = TensorBoardLogger(args.log_dir, name=args.name)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(monitor="loss", save_last=True, mode='min')

    trainer = Trainer(
        accelerator='gpu',
        # strategy='ddp',
        logger=logger,
        gpus=list(map(int, args.cuda.split(','))),
        max_epochs=args.epochs,
        # accumulate_grad_batches=args.acc_batches,
        callbacks=[lr_monitor, checkpoint_cb],
        check_val_every_n_epoch=10,
        log_every_n_steps=50,
        detect_anomaly=True,
        precision=16 if not args.predict else 32
    )
    return trainer


def build_datamodule(args):
    dm = RXNDataModule(
        root=args.dataset,
        batch_size=args.batch_size,
        fast_read=not args.not_fast_read,
        num_workers=args.num_workers,
        predict=args.predict or args.test,
        dataset_format=args.dataset_format,
        seed=args.seed,
    )
    return dm


def build_model(args):
    if args.model_path == '':
        model = RxnYd(
            d_model=args.d_model,
            nhead=args.nhead,
            p_layer=args.p_layer,
            r_layer=args.r_layer,
            output_layer=args.output_layer,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            known_rxn_cnt=args.known_rxn_cnt,
            norm_first=args.norm_first,
            activation=args.activation,
            weight_decay=args.weight_decay,
            use_3d_info=args.use_3d_info,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            max_single_hop=args.max_single_hop,
            use_dist_adj=not args.not_use_dist_adj,
            use_contrastive=not args.not_use_contrastive,
        )
    else:
        model = RxnYd.load_from_checkpoint(
            args.model_path,
            strict=True,
        )

    return model


if __name__ == '__main__':
    main()
