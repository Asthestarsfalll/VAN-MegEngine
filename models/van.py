import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import hub
from functools import partial
from .utils import DropPath, to_2tuple, init_weights


class DWConv(M.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = M.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(M.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=M.GELU,
        drop=0.
    ):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = M.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = M.Conv2d(hidden_features, out_features, 1)
        self.drop = M.Dropout(drop)
        self.apply(init_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(M.Module):
    def __init__(self, dim):
        super(LKA, self).__init__()
        self.conv0 = M.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = M.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = M.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return identity * attn


class Attention(M.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()

        self.proj_1 = M.Conv2d(d_model, d_model, 1)
        self.activation = M.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = M.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        identity = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + identity
        return x


class Block(M.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.,
        drop=0.,
        drop_path=0.,
        act_layer=M.GELU
    ):
        super(Block, self).__init__()
        self.norm1 = M.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else M.Identity()

        self.norm2 = M.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = mge.Parameter(
            layer_scale_init_value * F.ones((dim)))
        self.layer_scale_2 = mge.Parameter(
            layer_scale_init_value * F.ones((dim)))

        self.apply(init_weights)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.reshape(-1,
                               1, 1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.reshape(-1,
                               1, 1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(M.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=7,
        stride=4,
        in_chans=3,
        embed_dim=768
    ):
        super(OverlapPatchEmbed, self).__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = M.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                             padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = M.BatchNorm2d(embed_dim)

        self.apply(init_weights)

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[-2:]
        x = self.norm(x)
        return x, H, W


class VAN(M.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=M.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in F.linspace(0, drop_path_rate,
                                            sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = [Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])]
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = M.Linear(
            embed_dims[3], num_classes) if num_classes > 0 else M.Identity()

        self.apply(init_weights)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = M.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else M.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = F.flatten(x, 2).transpose((0, 2, 1))
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).transpose((0, 3, 1, 2))

        return x.mean(axis=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/38/files/80649661-c45f-44eb-84a7-cf6bd0d8b889"
)
def van_b0(**kwargs):
    model = VAN(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[3, 3, 5, 2],
        **kwargs)
    return model


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/38/files/923e3c3a-eb0b-4133-acb9-3a6f90d1753d"
)
def van_b1(**kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[2, 2, 4, 2],
        **kwargs)
    return model


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/38/files/5131b73d-63a4-47e9-907a-623157b9d9e7"
)
def van_b2(**kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[3, 3, 12, 3],
        **kwargs)
    return model


def van_b3(**kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[3, 5, 27, 3],
        **kwargs)
    return model


def van_b4(**kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3],
        **kwargs)
    return model


def van_b5(**kwargs):
    model = VAN(
        embed_dims=[96, 192, 480, 768], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[3, 3, 24, 3],
        **kwargs)
    return model


def van_b6(**kwargs):
    model = VAN(
        embed_dims=[96, 192, 384, 768], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(M.LayerNorm, eps=1e-6), depths=[6, 6, 90, 6],
        **kwargs)
    return model
