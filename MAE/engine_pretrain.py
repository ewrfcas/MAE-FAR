# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
import sys
from typing import Iterable

import cv2
import numpy as np
import torch

import MAE.util.lr_sched as lr_sched
import MAE.util.misc as misc


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    first = True

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # samples = samples.to(device, non_blocking=True)
        for k in samples:
            if type(samples[k]) is torch.Tensor:
                samples[k] = samples[k].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, pred, mask = model(samples['img'], samples['mask'], mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        if first and misc.is_main_process():
            first = False
            model_without_ddp = model.module if hasattr(model, 'module') else model
            os.makedirs(args.output_dir + '/samples', exist_ok=True)
            y = model_without_ddp.unpatchify(pred[:4])
            y = torch.einsum('nchw->nhwc', y).detach()

            # visualize the mask
            mask = mask[:4].detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, model_without_ddp.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
            mask = model_without_ddp.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('nchw->nhwc', mask).detach()  # .cpu()

            x = torch.einsum('nchw->nhwc', samples['img'][:4])

            # masked image
            im_masked = x * (1 - mask)
            im_masked = im_masked.cpu()
            im_masked = torch.cat(tuple(im_masked), dim=0)

            # MAE reconstruction pasted with visible patches
            im_paste = x * (1 - mask) + y * mask
            im_paste = im_paste.cpu()
            im_paste = torch.cat(tuple(im_paste), dim=0)
            x = x.cpu()
            y = y.cpu()
            x = torch.cat(tuple(x), dim=0)
            y = torch.cat(tuple(y), dim=0)

            images = torch.cat([x.float(), im_masked.float(), y.float(), im_paste.float()], dim=1)
            images = torch.clip((images * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255, 0, 255).int()
            images = images.numpy().astype(np.uint8)

            path = os.path.join(args.output_dir, 'samples')
            name = os.path.join(path, str(epoch).zfill(10) + ".jpg")
            print('\nsaving sample ' + name)
            cv2.imwrite(name, images[:, :, ::-1])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_finetune(model: torch.nn.Module,
                             data_loader: Iterable, optimizer, optimizer_new,
                             device: torch.device, epoch: int, loss_scaler,
                             log_writer=None, args=None, start_epoch=0):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('alpha', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    optimizer_new.zero_grad()
    first = True

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_finetune(optimizer, data_iter_step / len(data_loader) + epoch, args, start_from=start_epoch)
            lr_sched.adjust_learning_rate_finetune(optimizer_new, data_iter_step / len(data_loader) + epoch, args, start_from=start_epoch)

        # samples = samples.to(device, non_blocking=True)
        for k in samples:
            if type(samples[k]) is torch.Tensor:
                samples[k] = samples[k].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, pred, mask, irr_mask, partial_mask = model(samples['img'], samples['mask'], mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, optimizer_new, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            optimizer_new.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(alpha=model.module.alpha.item())

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('alpha', model.module.alpha.item(), epoch_1000x)

        if first and misc.is_main_process():
            first = False
            os.makedirs(args.output_dir + '/samples', exist_ok=True)
            model_without_ddp = model.module if hasattr(model, 'module') else model
            y = model_without_ddp.unpatchify(pred[:4])
            y = torch.einsum('nchw->nhwc', y).detach()

            # visualize the mask
            mask = mask[:4].detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, model_without_ddp.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
            mask = model_without_ddp.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('nchw->nhwc', mask).detach()  # .cpu()

            partial_mask = partial_mask[:4].detach()
            partial_mask = partial_mask.repeat(1, 1, model_without_ddp.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
            partial_mask = model_without_ddp.unpatchify(partial_mask)  # 1 is removing, 0 is keeping
            partial_mask = torch.einsum('nchw->nhwc', partial_mask).detach()  # .cpu()

            irr_mask = irr_mask[:8].detach()
            irr_mask = torch.einsum('nchw->nhwc', irr_mask).detach()  # .cpu()

            x = torch.einsum('nchw->nhwc', samples['img'][:8])

            # masked image
            im_masked = x * (1 - mask)
            im_masked = im_masked.cpu()
            im_masked = torch.cat(tuple(im_masked), dim=0)

            im_masked2 = x * (1 - irr_mask)
            im_masked2 = im_masked2.cpu()
            im_masked2 = torch.cat(tuple(im_masked2), dim=0)

            im_masked3 = x * (1 - partial_mask)
            im_masked3 = im_masked3.cpu()
            im_masked3 = torch.cat(tuple(im_masked3), dim=0)

            # MAE reconstruction pasted with visible patches
            im_paste = x * (1 - mask) + y * mask
            im_paste = im_paste.cpu()
            im_paste = torch.cat(tuple(im_paste), dim=0)
            x = x.cpu()
            y = y.cpu()
            x = torch.cat(tuple(x), dim=0)
            y = torch.cat(tuple(y), dim=0)

            images = torch.cat([x.float(), im_masked.float(), im_masked2.float(), im_masked3.float(), y.float(), im_paste.float()], dim=1)
            images = torch.clip((images * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255, 0, 255).int()
            images = images.numpy().astype(np.uint8)

            path = os.path.join(args.output_dir, 'samples')
            name = os.path.join(path, str(epoch).zfill(10) + ".jpg")
            print('\nsaving sample ' + name)
            cv2.imwrite(name, images[:, :, ::-1])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
