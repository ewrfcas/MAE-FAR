import argparse

import torch.nn.parallel
from pytorch_lightning import Trainer, loggers
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint

from ACR.base.parse_config import ConfigParser
from ACR.networks.discriminators import NLayerDiscriminator
from ACR.networks.generators import ACRModel
from ACR.trainer import PLTrainer
from MAE.util.misc import get_mae_model
import numpy as np
import random


def main(args, config):
    checkpoint_callback = ModelCheckpoint(f'ckpts/{args.exp_name}/models', monitor='val/FID', mode='min', save_top_k=1, save_last=True)

    logger = loggers.TestTubeLogger(
        save_dir="ckpts",
        name=args.exp_name,
        version=0,
        debug=False,
        create_git_tag=False
    )

    # build models architecture, then print to console
    mae = get_mae_model('mae_vit_base_patch16', mask_decoder=config['mask_decoder'])
    acr = ACRModel(config['g_args'])
    D = NLayerDiscriminator(config['d_args'])

    model = PLTrainer(mae, acr, D, config, 'ckpts/' + args.exp_name, num_gpus=args.num_gpus, use_ema=args.use_ema, dynamic_size=False)

    print("Loading checkpoint: {} ...".format(args.resume_mae))
    checkpoint = torch.load(args.resume_mae, map_location='cpu')
    model.mae.load_state_dict(checkpoint['model'])

    if args.use_ema:
        model.reset_ema()

    trainer = Trainer(max_steps=config['trainer']['total_step'],
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=args.pl_resume,
                      logger=logger,
                      progress_bar_refresh_rate=1,
                      gpus=-1,
                      log_every_n_steps=config['trainer']['logging_every'],
                      num_sanity_val_steps=-1,  # set val test before the training
                      val_check_interval=config['trainer']['eval_period'],
                      benchmark=True,
                      accelerator='ddp' if args.num_gpus > 1 else None,
                      terminate_on_nan=False,
                      precision=32)

    trainer.fit(model)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-e', '--exp_name', default=None, type=str)
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('--use_ema', action='store_true', help='Whether to use ema?')
    args.add_argument('--resume_mae', default=None, type=str, help='MAE path to restore')

    # custom cli options to modify configuration from default values given in json file.
    args = args.parse_args()
    config = ConfigParser.from_args(args, mkdir=True)
    SEED = 123
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    num_gpus = torch.cuda.device_count()
    args.num_gpus = num_gpus

    main(args, config)
