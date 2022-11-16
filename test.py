import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.parallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from ACR.base.dataset import DynamicFARDataset
from ACR.base.parse_config import ConfigParser
from ACR.inpainting_metric import get_inpainting_metrics
from ACR.networks.generators import ACRModel
from MAE.util.misc import get_mae_model


def main(gpu, args, config):
    rank = 0
    torch.cuda.set_device(gpu)

    # dataloader
    data_args = config['dataset']
    data_args['rel_pos_num'] = config['g_args']['rel_pos_num']
    data_args['use_mpe'] = config['g_args']['use_mpe']
    data_args['default_size'] = args.image_size
    val_dataset = DynamicFARDataset(data_args, config['val_flist'], mask_path=None,
                                    batch_size=config['batch_size'], augment=False, training=False,
                                    test_mask_path=config['test_mask_flist'], world_size=1)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True, batch_size=config['batch_size'], num_workers=4)

    # build models architecture, then print to console
    acr = ACRModel(config['g_args']).requires_grad_(False)
    acr.cuda(gpu).eval()
    mae = get_mae_model('mae_vit_base_patch16', mask_decoder=config['mask_decoder']).requires_grad_(False)
    mae.cuda(gpu).eval()
    if rank == 0:
        print('MAE', sum(p.numel() for p in mae.parameters()))
        print('G', sum(p.numel() for p in acr.G.parameters()))
        print('GCs', sum(p.numel() for p in acr.GCs.parameters()))

    if args.load_pl:  # load ckpt from pytorch lightning
        print("Loading checkpoint: {} ...".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')['state_dict']
        mae_weights = {}
        g_weights = {}
        for k in checkpoint:
            if k.startswith('mae.'):
                mae_weights[k.replace('mae.', '')] = checkpoint[k]
            if k.startswith('acr.'):
                g_weights[k.replace('acr.', '')] = checkpoint[k]
        mae.load_state_dict(mae_weights)
        acr.load_state_dict(g_weights)
    else:
        print("Loading checkpoint: {} ...".format(args.mae_ckpt))
        checkpoint = torch.load(args.mae_ckpt, map_location='cpu')
        mae.load_state_dict(checkpoint['model'])
        if args.g_ckpt is None:
            args.g_ckpt = 'G_last.pth'
        resume_path = os.path.join(str(config.resume), args.g_ckpt)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')
        acr.G.load_state_dict(checkpoint['generator'])
        acr.GCs.load_state_dict(checkpoint['gc_encoder'])

    eval_path = args.output_path
    os.makedirs(eval_path, exist_ok=True)

    with torch.no_grad():
        for items in tqdm(val_loader):
            for k in items:
                if type(items[k]) is torch.Tensor:
                    items[k] = items[k].cuda()
            # mae inference
            mae_feats, scores = mae.forward_return_feature(items['img_256'], items['mask_256'])
            items['mae_feats'] = mae_feats
            items['scores'] = scores

            # model inference
            gen_img = acr.forward(items)
            gen_img = items['image'] * (1 - items['mask']) + gen_img * items['mask']
            gen_img = torch.clamp(gen_img * 255.0, 0, 255)
            gen_img = gen_img.permute(0, 2, 3, 1).int().cpu().numpy()
            for img_num in range(items['image'].shape[0]):
                cv2.imwrite(os.path.join(eval_path, items['name'][img_num]), gen_img[img_num, :, :, ::-1])

        metric = get_inpainting_metrics(eval_path, config['test_path'])
        print(metric)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-e', '--exp_name', default=None, type=str)
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('--mae_ckpt', default=None, type=str)
    args.add_argument('--g_ckpt', default=None, type=str)
    args.add_argument('--output_path', type=str, default='./outputs')
    args.add_argument('--load_pl', action='store_true')
    args.add_argument('--image_size', type=int, default=256, help='Test image size')

    # custom cli options to modify configuration from default values given in json file.
    args = args.parse_args()
    config = ConfigParser.from_args(args, mkdir=False)
    SEED = 3407
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cudnn.benchmark = True

    args.world_size = 1

    main(0, args, config)
