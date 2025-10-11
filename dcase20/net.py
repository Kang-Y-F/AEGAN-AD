import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def ActLayer(act):
    assert act in ['relu', 'leakyrelu', 'tanh'], 'Unknown activate function!'
    if act == 'relu':
        return nn.ReLU(True)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.2, True)
    elif act == 'tanh':
        return nn.Tanh()


def NormLayer(normalize, chan, reso):
    assert normalize in ['bn', 'ln', 'in'], 'Unknown normalize function!'
    if normalize == 'bn':
        return nn.BatchNorm2d(chan)
    elif normalize == 'ln':
        return nn.LayerNorm((chan, reso, reso))
    elif normalize == 'in':
        return nn.InstanceNorm2d(chan)


class DCEncoder(nn.Module):
    """
    DCGAN DCEncoder NETWORK
    """

    def __init__(self, isize, nz, ndf, act, normalize, add_final_conv=True):
        super(DCEncoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = []
        main.append(nn.Conv2d(1, ndf, 4, 2, 1, bias=False))
        main.append(NormLayer(normalize, ndf, isize // 2))
        main.append(ActLayer(act))
        csize, cndf = isize // 2, ndf

        while csize > 4:
            in_chan = cndf
            out_chan = cndf * 2
            main.append(nn.Conv2d(in_chan, out_chan, 4, 2, 1, bias=False))
            cndf = cndf * 2
            csize = csize // 2
            main.append(NormLayer(normalize, out_chan, csize))
            main.append(ActLayer(act))

        # state size. K x 4 x 4
        if add_final_conv:
            main.append(nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        self.main = nn.Sequential(*main)

    def forward(self, x):
        z = self.main(x)
        return z


class DCDecoder(nn.Module):
    """
    DCGAN DCDecoder NETWORK
    """
    def __init__(self, isize, nz, ngf, act, normalize):
        super(DCDecoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = []
        main.append(nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        csize = 4
        main.append(NormLayer(normalize, cngf, csize))
        main.append(ActLayer(act))

        while csize < isize // 2:
            main.append(nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            cngf = cngf // 2
            csize = csize * 2
            main.append(NormLayer(normalize, cngf, csize))
            main.append(ActLayer(act))

        main.append(nn.ConvTranspose2d(cngf, 1, 4, 2, 1, bias=False))
        main.append(ActLayer('tanh'))
        self.main = nn.Sequential(*main)

    def forward(self, z):
        x = self.main(z)
        return x


class AEDC(nn.Module):
    def __init__(self, param):
        super(AEDC, self).__init__()
        self.Encoder = DCEncoder(isize=param['net']['isize'],
                                 nz=param['net']['nz'],
                                 ndf=param['net']['ndf'],
                                 act=param['net']['act'][0],
                                 normalize=param['net']['normalize']['g'])
        self.Decoder = DCDecoder(isize=param['net']['isize'],
                                 nz=param['net']['nz'],
                                 ngf=param['net']['ngf'],
                                 act=param['net']['act'][1],
                                 normalize=param['net']['normalize']['g'])

    def forward(self, data, outz=False):
        z = self.Encoder(data)
        if outz:
            return z
        else:
            recon = self.Decoder(z)
            return recon


class Discriminator(nn.Module):
    def __init__(self, param):
        super(Discriminator, self).__init__()
        ndf, isize = param['net']['ndf'], param['net']['isize']
        act, normalize = param['net']['act'][0], param['net']['normalize']['d']
        assert isize % 16 == 0, "isize must be multiple of 16"

        blocks = []

        # block0: 1 -> ndf, H/2
        blocks.append(nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            NormLayer(normalize, ndf, isize // 2),
            ActLayer(act)
        ))

        csize, cndf = isize // 2, ndf
        # 继续下采样直到 4x4
        while csize > 4:
            in_c = cndf
            out_c = cndf * 2
            csize = csize // 2
            blocks.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                NormLayer(normalize, out_c, csize),
                ActLayer(act)
            ))
            cndf = out_c

        self.blocks = nn.ModuleList(blocks)

        # 深度可分离 4x4 -> 1x1，然后展平得到 D 的 embedding
        self.feat_extract_layer = nn.Sequential(
            nn.Conv2d(cndf, cndf, 4, 1, 0, bias=False, groups=cndf),  # depthwise conv to 1x1
            nn.Flatten()  # [B, cndf]
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(cndf),
            ActLayer(act),
            nn.Linear(cndf, 1)
        )

        # 记录：返回哪些层作为金字塔特征（这里取倒数第3/第2/最后一个卷积块的输出）
        self.pyramid_indices = list(range(max(0, len(self.blocks) - 3), len(self.blocks)))

    def forward(self, x, return_pyramid=False):
        feats = []
        h = x
        for li, block in enumerate(self.blocks):
            h = block(h)
            feats.append(h)

        # 判别 embedding + logit
        emb = self.feat_extract_layer(h)      # [B, C]
        logit = self.output_layer(emb)        # [B, 1]

        if return_pyramid:
            # 取中高层特征（分辨率较小、语义更强）作为金字塔
            pyr = [feats[i] for i in self.pyramid_indices]
            return logit, pyr
        else:
            return logit, h  # 兼容旧接口：返回最后一层特征


# ====== 时间步嵌入（Sinusoidal + MLP） ======
class TimeEmbedding(nn.Module):
    def __init__(self, dim=128, max_period=10000):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.max_period = max_period

    def forward(self, t):
        # t: [B] (long)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if emb.shape[1] < self.dim:  # 若 dim 为奇数，补一列 0
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.mlp(emb)  # [B, dim]

# ====== 条件特征投影：把 D 的多层特征降维+对齐到 x_t 的空间，然后拼接 ======
class CondProjector(nn.Module):
    def __init__(self, in_ch_list=None, out_total_ch=96):
        super().__init__()
        # 如果你知道各层通道数，可传 in_ch_list；不知道就 runtime 用 1x1 自动降到固定每层通道
        self.per_layer_out = out_total_ch // 3  # 假设三层
        self.proj = nn.ModuleList([
            nn.Conv2d(in_channels=self.per_layer_out, out_channels=self.per_layer_out, kernel_size=1)
        ])
        # 注：为通用起见，下面 forward 里会按实际通道动态建 1x1（简单起见）

    def forward(self, D_feats, target_hw=None):
        # D_feats: list[Tensor]，每个 [B, C_l, H_l, W_l]
        # 目标分辨率 = x_t 的分辨率（由调用方传入），或者用最大那个特征的
        if target_hw is None:
            # 取列表中分辨率最小的作为基准再上采样
            H = max([f.shape[-2] for f in D_feats])
            W = max([f.shape[-1] for f in D_feats])
            target_hw = (H, W)

        proj_outs = []
        for f in D_feats:
            B, C, H, W = f.shape
            # 动态 1x1 降到固定通道数
            conv1x1 = nn.Conv2d(C, self.per_layer_out, kernel_size=1, bias=False).to(f.device)
            f1 = conv1x1(f)
            f1 = F.interpolate(f1, size=target_hw, mode='bilinear', align_corners=False)
            proj_outs.append(f1)
        cond = torch.cat(proj_outs, dim=1)  # [B, out_total_ch, Ht, Wt]
        return cond

# ====== 极简 SmallUNet：输入 concat(x_t, cond, time_emb_map) 输出 eps_pred ======
class SmallUNet(nn.Module):
    def __init__(self, in_ch=1, cond_ch=96, t_dim=128, base_ch=64):
        super().__init__()
        self.t_to_affine = nn.Sequential(
            nn.Linear(t_dim, base_ch*2),
            nn.SiLU(),
            nn.Linear(base_ch*2, base_ch*2)
        )
        # encoder
        self.conv_in = nn.Conv2d(in_ch + cond_ch + 1, base_ch, 3, 1, 1)  # +1 给时间嵌入的平铺通道
        self.down = nn.Conv2d(base_ch, base_ch, 3, 1, 1)
        # decoder
        self.up = nn.Conv2d(base_ch, base_ch, 3, 1, 1)
        self.conv_out = nn.Conv2d(base_ch, in_ch, 3, 1, 1)

        self.act = nn.SiLU()
        self.norm1 = nn.BatchNorm2d(base_ch)
        self.norm2 = nn.BatchNorm2d(base_ch)

    def forward(self, x_t, t_emb, cond):
        B, _, H, W = x_t.shape
        # 把 time embedding 平铺成 [B,1,H,W] 一个通道（简单做法）
        t_map = t_emb.mean(dim=1, keepdim=True)  # [B,1]
        t_map = t_map.view(B, 1, 1, 1).expand(B, 1, H, W)

        h = torch.cat([x_t, cond, t_map], dim=1)  # [B, 1+cond_ch+1, H, W]
        h = self.conv_in(h)
        h = self.act(self.norm1(h))
        h = self.down(h)
        h = self.act(self.norm2(h))
        h = self.up(h)
        eps = self.conv_out(h)
        return eps

class DiffusionRefiner(nn.Module):
    def __init__(self, param):
        super().__init__()
        cond_ch_total = param['model'].get('D_cond_ch_total', 96)  # 建议在 config.yaml 设置
        t_dim = param['model'].get('t_emb_dim', 128)

        self.t_embed = TimeEmbedding(dim=t_dim)
        self.cond_proj = CondProjector(out_total_ch=cond_ch_total)
        self.unet = SmallUNet(in_ch=1, cond_ch=cond_ch_total, t_dim=t_dim, base_ch=64)

    def forward(self, x_t, t, D_feats):
        # D_feats 是 list[Tensor]，来自 netD(mel, return_pyramid=True)
        cond = self.cond_proj(D_feats, target_hw=x_t.shape[-2:])
        te = self.t_embed(t)  # [B, t_dim]
        eps_pred = self.unet(x_t, te, cond)
        return eps_pred



