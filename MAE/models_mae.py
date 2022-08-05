# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import numpy as np
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed

from MAE.util.pos_embed import get_2d_sincos_pos_embed
from MAE.vision_transformer import Block


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    Since [cls] is useless in inpainting, we remove it.
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 norm_pix_loss=False, init=True, random_mask=False, mask_decoder=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, mask_decoder=mask_decoder)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # encoder to decoder
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.random_mask = random_mask
        self.mask_decoder = mask_decoder

        if init:
            self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def adaptive_random_masking(self, x, mask, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        mask: [N, 1, 256, 256]
        """
        N, L, D = x.shape  # batch, length, dim
        s = int(np.sqrt(L))
        mask = F.interpolate(mask, size=[s, s], mode='area')
        mask[mask > 0] = 1  # [N,1,S,S]
        mask = mask.reshape(N, L)  # [N,L]

        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # make noise with mask in largest 1
        noise = torch.clamp(noise + mask, 0.0, 1.0)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        if self.random_mask:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            x, mask, ids_restore = self.adaptive_random_masking(x, mask, mask_ratio)

        # apply Transformer blocks
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        '''
        forward decoder during training needs ids_restore
        '''
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = x_

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x, _ = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_encoder_with_mask(self, x, mask):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed
        N, L, D = x.shape  # batch, length, dim
        s = int(np.sqrt(L))
        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        mask = F.interpolate(mask, size=[s, s], mode='area')
        mask_small = mask.clone()
        mask[mask > 0] = 1  # [N,1,S,S]
        mask_small[mask_small < 1] = 0
        mask = mask.reshape(N, L).unsqueeze(1).unsqueeze(1)  # [N,1,1,L]

        # apply Transformer blocks
        for blk in self.blocks:
            x, _ = blk(x, mask)
        x = self.norm(x)  # N,L,D
        mask = mask.squeeze(1).squeeze(1)  # N, L
        mask_small = mask_small.reshape(N, L).unsqueeze(1).unsqueeze(1) # [N,1,1,L]
        return x, mask, mask_small

    def forward_decoder_with_mask(self, x, mask):
        x = self.decoder_embed(x)  # N,L,D
        N, L, D = x.shape  # batch, length, dim
        mask = mask.unsqueeze(-1)  # N,L,1
        # append mask tokens to sequence
        x = x * (1 - mask) + self.mask_token.repeat(N, L, 1) * mask

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x, _ = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_decoder_return_feature(self, x, mask, mask_small):
        # embed tokens
        x = self.decoder_embed(x)  # N,L,D
        N, L, D = x.shape  # batch, length, dim
        mask = mask.unsqueeze(-1)  # N,L,1
        # append mask tokens to sequence
        x = x * (1 - mask) + self.mask_token.repeat(N, L, 1) * mask

        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        scores = []
        for blk in self.decoder_blocks:
            if self.mask_decoder:
                x, score = blk(x, mask_small)
            else:
                x, score = blk(x)
            scores.append(score.unsqueeze(1))
        scores = torch.mean(torch.cat(scores, dim=1), dim=1)  # [B,256,256]
        x = self.decoder_norm(x)
        return x, scores

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask, mask_ratio=0.75):
        """
        return loss, pred img, mask. Used during training.
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def forward_return_feature(self, imgs, mask):
        """
        return pred(feature), scores(attention). Used during finetuning.
        """
        latent, new_mask, mask_small = self.forward_encoder_with_mask(imgs, mask)
        pred, scores = self.forward_decoder_return_feature(latent, new_mask, mask_small)  # [N, L, D]
        N, L, D = pred.shape  # batch, length, dim
        s = int(np.sqrt(L))
        pred = pred.reshape(N, s, s, D).permute(0, 3, 1, 2)
        return pred, scores

    def forward_return_image(self, imgs, mask):
        """
        return Image, new_mask. Used during testing.
        """
        latent, new_mask, _ = self.forward_encoder_with_mask(imgs, mask)
        image = self.forward_decoder_with_mask(latent, new_mask)  # [N, L, D]
        image = self.unpatchify(image)
        return image, new_mask


class MaskedAutoencoderViTFinetune(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, random_mask=False, mask_decoder=False):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth,
                         num_heads=num_heads, decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
                         decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer, norm_pix_loss=norm_pix_loss,
                         random_mask=random_mask, mask_decoder=mask_decoder, init=False)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_patch_embed = PatchEmbed(img_size, patch_size, in_chans + 1, decoder_embed_dim)
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        w2 = self.decoder_patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def adaptive_random_masking(self, x, mask, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        mask: [N, 1, 256, 256]
        """
        irr_mask = mask
        N, L, D = x.shape  # batch, length, dim
        s = int(np.sqrt(L))
        mask = F.interpolate(mask, size=[s, s], mode='area')
        mask[mask > 0] = 1  # [N,1,S,S]
        mask = mask.reshape(N, L)  # [N,L]

        resize_mask = mask

        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # make noise with mask in largest 1
        noise = torch.clamp(noise + mask, 0.0, 1.0)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        resize_mask_com = (1 - resize_mask) * mask
        resize_mask_com = resize_mask_com.reshape(N, s, s).unsqueeze(1)
        resize_mask_com = F.interpolate(resize_mask_com, size=[256, 256], mode='nearest')
        resize_mask = resize_mask * mask
        resize_mask = resize_mask.reshape(N, s, s).unsqueeze(1)
        resize_mask = F.interpolate(resize_mask, size=[256, 256], mode='nearest')

        irr_mask = irr_mask * resize_mask + resize_mask_com

        return x_masked, mask, ids_restore, irr_mask

    def forward_encoder(self, x, mask, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, irr_mask = self.adaptive_random_masking(x, mask, mask_ratio)

        # apply Transformer blocks
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, irr_mask

    def forward_decoder(self, x, ids_restore, mask, imgs, irr_mask):
        '''
        forward decoder during training needs ids_restore
        '''
        # embed tokens
        x = self.decoder_embed(x)

        # img:[b,3,H,W] irr_mask:[b,1,H,W]
        input2embed = torch.cat([imgs * (1 - irr_mask), irr_mask], dim=1)
        partial_mask = F.interpolate(irr_mask, size=(16, 16), mode='area')  # [b,1,16,16]
        partial_mask = partial_mask.reshape(partial_mask.shape[0], 16 * 16, 1)  # [b,256,1]
        total_mask = partial_mask.clone()
        partial_mask[partial_mask < 1] = 0
        total_mask[total_mask > 0] = 1
        partial_mask = total_mask - partial_mask
        img_embedding = self.decoder_patch_embed(input2embed)  # [b,16,16,512]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        x = x_ * (1 - partial_mask) + (x_ * alpha + img_embedding * (1 - alpha)) * partial_mask

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x, _ = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x, partial_mask

    def forward_decoder_with_mask(self, x, mask, imgs, irr_mask):
        x = self.decoder_embed(x)  # N,L,D

        # img:[b,3,H,W] irr_mask:[b,1,H,W]
        input2embed = torch.cat([imgs * (1 - irr_mask), irr_mask], dim=1)
        partial_mask = F.interpolate(irr_mask, size=(16, 16), mode='area')  # [b,1,16,16]
        partial_mask = partial_mask.reshape(partial_mask.shape[0], 16 * 16, 1)  # [b,256,1]
        total_mask = partial_mask.clone()
        partial_mask[partial_mask < 1] = 0
        total_mask[total_mask > 0] = 1
        partial_mask = total_mask - partial_mask
        img_embedding = self.decoder_patch_embed(input2embed)  # [b,16,16,512]

        N, L, D = x.shape  # batch, length, dim
        mask = mask.unsqueeze(-1)  # N,L,1
        # append mask tokens to sequence
        x = x * (1 - mask) + self.mask_token.repeat(N, L, 1) * mask

        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        x = x * (1 - partial_mask) + (x * alpha + img_embedding * (1 - alpha)) * partial_mask

        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x, score = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_decoder_return_feature(self, x, mask, imgs, irr_mask, mask_small):
        # embed tokens
        x = self.decoder_embed(x)  # N,L,D

        # img:[b,3,H,W] irr_mask:[b,1,H,W]
        input2embed = torch.cat([imgs * (1 - irr_mask), irr_mask], dim=1)
        partial_mask = F.interpolate(irr_mask, size=(16, 16), mode='area')  # [b,1,16,16]
        partial_mask = partial_mask.reshape(partial_mask.shape[0], 16 * 16, 1)  # [b,256,1]
        total_mask = partial_mask.clone()
        partial_mask[partial_mask < 1] = 0
        total_mask[total_mask > 0] = 1
        partial_mask = total_mask - partial_mask
        img_embedding = self.decoder_patch_embed(input2embed)  # [b,16,16,512]

        N, L, D = x.shape  # batch, length, dim
        mask = mask.unsqueeze(-1)  # N,L,1
        # append mask tokens to sequence
        x = x * (1 - mask) + self.mask_token.repeat(N, L, 1) * mask

        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        x = x * (1 - partial_mask) + (x * alpha + img_embedding * (1 - alpha)) * partial_mask

        # add pos embed
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        scores = []
        for blk in self.decoder_blocks:
            if self.mask_decoder:
                x, score = blk(x, mask_small)
            else:
                x, score = blk(x)
            scores.append(score.unsqueeze(1))
        scores = torch.mean(torch.cat(scores, dim=1), dim=1)  # [B,256,256]
        x = self.decoder_norm(x)
        return x, scores

    def forward_loss(self, imgs, pred, irr_mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)

        irr_mask = irr_mask.repeat(1, 3, 1, 1)
        mask = self.patchify(irr_mask)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # should not get mean loss for ft

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask, mask_ratio=0.75):
        """
        return loss, pred img, mask. Used during training.
        """
        latent, mask, ids_restore, irr_mask = self.forward_encoder(imgs, mask, mask_ratio)
        pred, partial_mask = self.forward_decoder(latent, ids_restore, mask, imgs, irr_mask)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, irr_mask)
        return loss, pred, mask, irr_mask, partial_mask

    def forward_return_feature(self, imgs, mask):
        """
        return pred(feature), scores(attention). Used during finetuning.
        """
        latent, new_mask, mask_small = self.forward_encoder_with_mask(imgs, mask)
        pred, scores = self.forward_decoder_return_feature(latent, new_mask, imgs, mask, mask_small)  # [N, L, D]
        N, L, D = pred.shape  # batch, length, dim
        s = int(np.sqrt(L))
        pred = pred.reshape(N, s, s, D).permute(0, 3, 1, 2)
        return pred, scores

    def forward_return_image(self, imgs, mask):
        """
        return Image, new_mask. Used during testing.
        """
        latent, new_mask, _ = self.forward_encoder_with_mask(imgs, mask)
        image = self.forward_decoder_with_mask(latent, new_mask, imgs, mask)  # [N, L, D]
        image = self.unpatchify(image)
        return image, new_mask
