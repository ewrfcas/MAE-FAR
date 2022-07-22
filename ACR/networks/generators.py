import torch.nn.functional as F

from ACR.networks.attention import GroupConvAttention
from ACR.networks.ffc import FFCResnetBlock
from ACR.networks.layers import *


class FFCGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.group_attn = GroupConvAttention()

        # resnet blocks
        blocks = []
        for i in range(9):
            blocks.append(FFCResnetBlock(512, 1))

        self.middle = nn.Sequential(*blocks)

        self.convt1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt1 = nn.BatchNorm2d(256)

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt2 = nn.BatchNorm2d(128)

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bnt3 = nn.BatchNorm2d(64)

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)
        self.act_last = nn.Sigmoid()

    def forward(self, x, prior_feats, rel_pos_emb=None, direct_emb=None, scores=None, alpha_att1=None, alpha_att2=None):
        x = self.pad1(x)
        x = self.conv1(x)
        if self.config['use_mpe']:
            inp = x.to(torch.float32) + rel_pos_emb + direct_emb
        else:
            inp = x.to(torch.float32)
        x = self.bn1(inp)
        x = self.act(x)

        x = self.conv2(x + prior_feats[0])
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv3(x + prior_feats[1])
        x = self.bn3(x)
        x = self.act(x)

        x = self.conv4(x + prior_feats[2])
        x = self.bn4(x)
        x = self.act(x)

        if self.config['use_attn']:
            x = self.group_attn(x, scores, alpha_att1)
        x = self.middle(x + prior_feats[3])
        if self.config['use_attn']:
            x = self.group_attn(x, scores, alpha_att2)

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)
        x = self.act_last(x)

        return x


class GCEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.act = nn.ReLU(True)

        self.convt0 = GateConv(512 + 2, 512, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt0 = nn.BatchNorm2d(512)
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt1 = GateConv(512, 256, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt1 = nn.BatchNorm2d(256)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt2 = GateConv(256, 128, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt2 = nn.BatchNorm2d(128)
        self.alpha3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt3 = GateConv(128, 64, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt3 = nn.BatchNorm2d(64)
        self.alpha4 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        if self.config['use_attn']:
            self.alpha_att1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
            self.alpha_att2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        if self.config['use_mpe']:
            self.rel_pos_emb = MaskedSinusoidalPositionalEmbedding(num_embeddings=config['rel_pos_num'], embedding_dim=64)
            self.direct_emb = MultiLabelEmbedding(num_positions=4, embedding_dim=64)
            self.alpha5 = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)
            self.alpha6 = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)

    def make_coord(self, shape, ranges=None, flatten=True):
        """ Make coordinates at grid centers.
        """
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def implicit_upsample(self, feat, H, W):
        feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
        [B, _, _, _] = feat.shape
        # [B,2,h,w]
        feat_coord = self.make_coord([H, W], flatten=False).to(feat.device).permute(2, 0, 1)
        feat_coord = feat_coord.unsqueeze(0).expand(B, 2, H, W).to(feat.dtype)
        feat = torch.cat([feat, feat_coord], dim=1)
        return feat

    def forward(self, x, rel_pos=None, direct=None, feat_size=None):
        x = self.implicit_upsample(x, int(feat_size[0]), int(feat_size[0]))

        x = self.convt0(x)
        x = self.bnt0(x.to(torch.float32))
        x = self.act(x)

        return_feats = [x * self.alpha1]

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(x * self.alpha2)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(x * self.alpha3)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(x * self.alpha4)

        return_feats = return_feats[::-1]

        meta = {'prior_feats': return_feats, 'alpha_att1': None, 'alpha_att2': None, 'rel_pos_emb': None, 'direct_emb': None}
        if self.config['use_attn']:
            meta['alpha_att1'] = self.alpha_att1
            meta['alpha_att2'] = self.alpha_att2

        if self.config['use_mpe']:
            b, h, w = rel_pos.shape
            rel_pos = rel_pos.reshape(b, h * w)
            rel_pos_emb = self.rel_pos_emb(rel_pos).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha5
            direct = direct.reshape(b, h * w, 4).to(torch.float32)
            direct_emb = self.direct_emb(direct).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha6
            meta['rel_pos_emb'] = rel_pos_emb
            meta['direct_emb'] = direct_emb

        return meta


class ACRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.G = FFCGenerator(config)
        self.GCs = GCEncoder(config)

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']
        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)

        scores = batch['scores'].detach()
        mae_feats = batch['mae_feats'].detach()
        # # scores:[B,256,256]
        # mask_16x16 = F.interpolate(batch['mask_256'], (16, 16), mode='area').squeeze(1)
        # mask_16x16[mask_16x16 < 1] = 0  # [B,16,16]
        # B = mask_16x16.shape[0]
        # mask_16x16 = mask_16x16.reshape(B, 1, 256)
        # scores = scores * (1 - mask_16x16)
        # scores = F.softmax(scores, dim=-1)  # remask and re-norm
        # mae_feats = batch['mae_feats'].detach().contiguous().to(torch.float32)

        if self.config['use_mpe']:
            meta = self.GCs(mae_feats, batch['rel_pos'], batch['direct'], batch['feat_size'])
        else:
            meta = self.GCs(mae_feats)
        gen_img = self.G(masked_img, meta['prior_feats'], meta['rel_pos_emb'], meta['direct_emb'],
                         scores, meta['alpha_att1'], meta['alpha_att2'])
        return gen_img
