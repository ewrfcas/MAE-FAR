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

    model = PLTrainer(mae, acr, D, config, 'ckpts/' + args.exp_name, num_gpus=args.num_gpus,
                      use_ema=args.use_ema, dynamic_size=args.dynamic_size)

    # init ckpt for finetuning
    if args.finetune_from_old:
        print("Loading checkpoint: {} ...".format(args.resume_mae))
        checkpoint = torch.load(args.resume_mae, map_location='cpu')
        model.mae.load_state_dict(checkpoint['model'])

        print("Loading checkpoint: {} ...".format(args.resume_G))
        checkpoint = torch.load(args.resume_G, map_location='cpu')
        model.acr.G.load_state_dict(checkpoint['generator'])
        model.acr.GCs.load_state_dict(checkpoint['gc_encoder'])
        model.g_opt_state = checkpoint['acr_opt']
        model.g_opt_state['param_groups'][0]['lr'] = config['optimizer']['g_opt']['lr']
        model.g_opt_state['param_groups'][0]['initial_lr'] = config['optimizer']['g_opt']['lr']

        print("Loading checkpoint: {} ...".format(args.resume_D))
        checkpoint = torch.load(args.resume_D, map_location='cpu')
        model.D.load_state_dict(checkpoint['discriminator'])
        model.d_opt_state = checkpoint['d_opt']
        model.d_opt_state['param_groups'][0]['lr'] = config['optimizer']['d_opt']['lr']
        model.d_opt_state['param_groups'][0]['initial_lr'] = config['optimizer']['d_opt']['lr']

    if args.use_ema:
        model.reset_ema()

    trainer = Trainer(max_steps=config['trainer']['total_step'],
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=args.pl_resume,
                      logger=logger,
                      progress_bar_refresh_rate=1,
                      gpus=-1,
                      log_every_n_steps=config['trainer']['logging_every'],
                      num_sanity_val_steps=0,  # set val test before the training (0 or -1)
                      val_check_interval=config['trainer']['eval_period'],
                      benchmark=True if not args.dynamic_size else False,
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
    args.add_argument('--dynamic_size', action='store_true', help='Whether to finetune in dynamic size?')
    args.add_argument('--use_ema', action='store_true', help='Whether to use ema?')
    args.add_argument('--finetune_from_old', action='store_true',
                      help='Whether to finetune from old weights which are not trained with PL(pytorch-lightning).')
    args.add_argument('--pl_resume', default=None, type=str, help='PL path to restore')
    args.add_argument('--resume_mae', default=None, type=str, help='MAE path to restore, only used for old codes')
    args.add_argument('--resume_G', default=None, type=str, help='Only used for old codes')
    args.add_argument('--resume_D', default=None, type=str, help='Only used for old codes')

    # custom cli options to modify configuration from default values given in json file.
    args = args.parse_args()
    config = ConfigParser.from_args(args, mkdir=True)
    SEED = 456
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    num_gpus = torch.cuda.device_count()
    args.num_gpus = num_gpus

    main(args, config)
