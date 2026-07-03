from typing import Dict
import typing as t
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(in_channels // reduction_ratio, in_channels)
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        # return x * self.sigmoid(y)
        max_out = self.fc(self.max_pool(x))
        avg_out = self.fc(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        return channel_out * x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class MSFANet2(nn.Module):
    # Multi-Scale Fusion Attention Network
    def __init__(self, in_channels: int, kernels=(1,3,5,7)):
        super().__init__()
        self.in_channels = in_channels
        self.kernels = list(kernels)
        self.num_branches = len(kernels)
        self.fuse_conv = nn.Conv2d(self.num_branches * in_channels, in_channels, kernel_size=1, bias=False)

        # ----- Multi-scale depthwise conv branches -----
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k//2, groups=in_channels, bias=False)
            for k in self.kernels
        ])
        self.branch_bn = nn.ModuleList([nn.BatchNorm2d(in_channels) for _ in self.kernels])
        self.branch_act = nn.ModuleList([nn.LeakyReLU(inplace=True) for _ in self.kernels])

        # ----- Two-stage attention -----
        self.attn = MCA(in_channels)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Self-Attention enriched feature
        x_sa = x

        # 2) Multi-scale depthwise branches
        branch_feats = []
        for conv, bn, act in zip(self.branches, self.branch_bn, self.branch_act):
            f = act(bn(conv(x_sa)))
            branch_feats.append(f)  # each: (B, C, H, W)

        # 3) 拼接分支
        cat = torch.cat(branch_feats, dim=1)  # [B, N*C, H, W]

        # 4) 用 1×1 卷积做通道融合，替代均值池化
        B, _, H, W = cat.shape
        cat = cat.view(B, self.num_branches, self.in_channels, H, W)  # [B, N, C, H, W]
        mixed = cat.mean(dim=1) # [B, C, H, W]

        # 5) Two-stage attention
        out = self.attn(mixed)


        # 残差增强
        fused = sum(branch_feats) / self.num_branches
        return out + fused

class KSFA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(KSFA, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # 扩张卷积用于多尺度感受野
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, padding=3, dilation=3)

        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 最终融合
        self.fusion = nn.Conv2d((in_channels // reduction) * 3, in_channels, kernel_size=1)

    def forward(self, x):
        # 多尺度特征提取
        scale1 = self.conv1(x)
        scale2 = self.conv2(x)
        scale3 = self.conv3(x)

        # 特征融合
        combined = torch.cat([scale1, scale2, scale3], dim=1)
        fused = self.fusion(combined)

        # 通道注意力
        attention = self.channel_attention(fused)
        output = fused * attention

        return output

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out

class CrossChannelCoordinateAttentionV2(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CrossChannelCoordinateAttentionV2, self).__init__()
        mip = max(8, in_channels // reduction)

        # 先做一次 1x1 卷积，把不同通道信息交互
        self.inter_channel_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 坐标注意力核心
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 先通道交互
        x = self.inter_channel_conv(x)

        # 坐标池化
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 拼接后压缩
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 再拆分
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 还原
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out

class MCA(nn.Module):
    def __init__(self, in_channels, kernels=(1,3,5,7), reduction=32):
        super().__init__()
        self.in_channels = in_channels
        self.kernels = kernels
        self.num_branches = len(kernels)
        mip = max(8, in_channels // reduction)

        self.convs_h = nn.ModuleList([
            nn.Conv2d(in_channels, mip, kernel_size=(k,1), padding=(k//2,0), bias=False)
            for k in kernels
        ])
        self.convs_w = nn.ModuleList([
            nn.Conv2d(in_channels, mip, kernel_size=(1,k), padding=(0,k//2), bias=False)
            for k in kernels
        ])

        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(mip),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.size()

        # 全局池化
        x_h = torch.mean(x, dim=3, keepdim=True)  # [B,C,H,1]
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0,1,3,2)  # [B,C,W,1] → [B,C,1,W]

        # 多尺度 h 分支
        feat_h = [conv(x_h) for conv in self.convs_h]   # 每个 [B,mip,H,1]
        cat_h = torch.cat(feat_h, dim=1)
        cat_h = cat_h.view(B, self.num_branches, -1, H, 1).mean(dim=1)
        cat_h = self.bn_act(cat_h)

        # 多尺度 w 分支
        feat_w = [conv(x_w) for conv in self.convs_w]   # 每个 [B,mip,W,1]
        cat_w = torch.cat(feat_w, dim=1)
        cat_w = cat_w.view(B, self.num_branches, -1, W, 1).mean(dim=1)
        cat_w = self.bn_act(cat_w)

        # -------- 用通道平均池化代替 1×1 卷积 --------
        # 对 mip 维度平均，然后扩展到 C 通道
        a_h = torch.sigmoid(torch.mean(cat_h, dim=1, keepdim=True))  # [B,1,H,1]
        a_h = a_h.expand(B, C, H, 1)

        a_w = torch.sigmoid(torch.mean(cat_w, dim=1, keepdim=True))  # [B,1,W,1]
        a_w = a_w.permute(0,1,3,2).expand(B, C, 1, W)

        out = x * a_h * a_w
        return out

class LGA(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(LGA, self).__init__()
        # Local Branch: Depthwise Conv for local context
        self.local_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.local_bn = nn.BatchNorm2d(channels)

        # Global Branch: Squeeze via global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Fusion Conv
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Local attention branch
        local_feat = self.local_bn(self.local_conv(x))

        # Global attention branch
        global_weight = self.global_fc(self.global_pool(x))
        global_feat = x * global_weight

        # Concatenate and fuse
        concat = torch.cat([local_feat, global_feat], dim=1)
        out = self.fusion(concat)
        return out

class SCSA(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA, self).__init__()  # 调用 nn.Module 的构造函数
        self.dim = dim  # 特征维度
        self.head_num = head_num  # 注意力头数
        self.head_dim = dim // head_num  # 每个头的维度
        self.scaler = self.head_dim ** -0.5  # 缩放因子
        self.group_kernel_sizes = group_kernel_sizes  # 分组卷积核大小
        self.window_size = window_size  # 窗口大小
        self.qkv_bias = qkv_bias  # 是否使用偏置
        self.fuse_bn = fuse_bn  # 是否融合批归一化
        self.down_sample_mode = down_sample_mode  # 下采样模式

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'  # 确保维度可被4整除
        self.group_chans = group_chans = self.dim // 4  # 分组通道数

        # 定义局部和全局深度卷积层
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)

        # 注意力门控层
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)  # 水平方向的归一化
        self.norm_w = nn.GroupNorm(4, dim)  # 垂直方向的归一化

        self.conv_d = nn.Identity()  # 直接连接
        self.norm = nn.GroupNorm(1, dim)  # 通道归一化
        # 定义查询、键和值的卷积层
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # 注意力丢弃层
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()  # 通道注意力门控

        # 根据窗口大小和下采样模式选择下采样函数
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans  # 重组合下采样
                # 维度降低
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)  # 平均池化
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)  # 最大池化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入张量 x 的维度为 (B, C, H, W)
        """
        # 计算空间注意力优先级
        b, c, h_, w_ = x.size()  # 获取输入的形状
        # (B, C, H)
        x_h = x.mean(dim=3)  # 沿着宽度维度求平均
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)  # 拆分通道
        # (B, C, W)
        x_w = x.mean(dim=2)  # 沿着高度维度求平均
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)  # 拆分通道

        # 计算水平注意力
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)  # 调整形状

        # 计算垂直注意力
        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)  # 调整形状

        # 计算最终的注意力加权
        x = x * x_h_attn * x_w_attn

        # 基于自注意力的通道注意力
        # 减少计算量
        y = self.down_func(x)  # 下采样
        y = self.conv_d(y)  # 维度转换
        _, _, h_, w_ = y.size()  # 获取形状

        # 先归一化，然后重塑 -> (B, H, W, C) -> (B, C, H * W)，并生成 q, k 和 v
        y = self.norm(y)  # 归一化
        q = self.q(y)  # 计算查询
        k = self.k(y)  # 计算键
        v = self.v(y)  # 计算值
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # 计算注意力
        attn = q @ k.transpose(-2, -1) * self.scaler  # 点积注意力计算
        attn = self.attn_drop(attn.softmax(dim=-1))  # 应用注意力丢弃
        # (B, head_num, head_dim, N)
        attn = attn @ v  # 加权值
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)  # 求平均
        attn = self.ca_gate(attn)  # 应用通道注意力门控
        return attn * x  # 返回加权后的输入

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class AblationAttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        reduction=16,
        groups=8,

        # ===== Ablation Switch =====
        use_ms=False,      # Multi-scale
        use_ca=True,      # Channel Attention
        use_sa=False,      # Spatial Attention
        use_res=True      # Residual
    ):
        super().__init__()

        self.use_ms = use_ms
        self.use_ca = use_ca
        self.use_sa = use_sa
        self.use_res = use_res

        # ======================================
        # Multi-scale Feature Extraction
        # ======================================

        self.conv_d1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            dilation=1,
            groups=groups,
            bias=False
        )

        if self.use_ms:
            self.conv_d2 = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=groups,
                bias=False
            )

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(inplace=True)

        # ======================================
        # Channel Attention
        # ======================================

        if self.use_ca:
            self.ca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),

                nn.Conv2d(
                    channels,
                    channels // reduction,
                    kernel_size=1,
                    bias=False
                ),

                nn.ReLU(inplace=True),

                nn.Conv2d(
                    channels // reduction,
                    channels,
                    kernel_size=1,
                    bias=False
                ),

                nn.Sigmoid()
            )

        # ======================================
        # Spatial Attention
        # ======================================

        if self.use_sa:

            self.conv_h = nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                groups=groups,
                bias=False
            )

            self.conv_w = nn.Conv2d(
                channels,
                channels,
                kernel_size=(3, 1),
                padding=(1, 0),
                groups=groups,
                bias=False
            )

            self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        identity = x

        # ======================================
        # Multi-scale
        # ======================================

        f1 = self.conv_d1(x)

        if self.use_ms:
            f2 = self.conv_d2(x)
            f = f1 + f2
        else:
            f = f1

        f = self.act(self.bn(f))

        # ======================================
        # Channel Attention
        # ======================================

        if self.use_ca:
            ca = self.ca(f)
        else:
            ca = 1.0

        # ======================================
        # Spatial Attention
        # ======================================

        if self.use_sa:

            h = self.conv_h(f)
            w = self.conv_w(f)

            sa = self.sigmoid(h + w)

        else:
            sa = 1.0

        # ======================================
        # Fusion
        # ======================================

        out = f * ca * sa

        # ======================================
        # Residual
        # ======================================

        if self.use_res:
            out = out + identity

        return out

class FastMKSFA(nn.Module):
    def __init__(self, channels, reduction=16, groups=8):
        super().__init__()

        # === Multi-scale (dilated) ===
        self.conv_d1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1, groups=groups, bias=False)
        self.conv_d2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=groups, bias=False)

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(inplace=True)

        # === Channel Attention ===
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # === Lightweight Coordinate Attention ===
        self.conv_h = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=groups, bias=False)
        self.conv_w = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=groups, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        # multi-scale (共享特征)
        f = self.conv_d1(x) + self.conv_d2(x)
        f = self.act(self.bn(f))

        # channel attention
        ca = self.ca(f)

        # spatial (direction-aware)
        h = self.conv_h(f)
        w = self.conv_w(f)

        attn = self.sigmoid(h + w)

        out = f * ca * attn

        return out + identity

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, use_attn = True):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.use_attn = use_attn
        if use_attn:
            self.attn = FastMKSFA(out_channels)
            # self.attn = CoordinateAttention(out_channels)
            # self.attn = ECA(out_channels)
            # self.attn = AblationAttentionBlock(out_channels)
            # self.attn = MultiScaleCoordinateAttention(out_channels)


    def forward(self, x):
        x = self.double_conv(x)
        if self.use_attn:
            x = self.attn(x)
        return x

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_attn = True):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels, use_attn = use_attn)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, use_attn = False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, use_attn = use_attn)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 4,
                 num_classes: int = 4,
                 bilinear: bool = True,
                 base_c: int = 32):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2, use_attn=True)   #channels:32→64
        self.down2 = Down(base_c * 2, base_c * 4, use_attn=True)   #channels:64→128
        self.down3 = Down(base_c * 4, base_c * 8, use_attn=True)   #channels:128→256
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor, use_attn=False)    #channels:256→256
        self.up1 = Up(base_c * 16, base_c * 8 // factor,bilinear, use_attn=False)
        self.up2 = Up(base_c * 8, base_c * 4 // factor,bilinear, use_attn=False)
        self.up3 = Up(base_c * 4, base_c * 2 // factor,bilinear, use_attn=False)
        self.up4 = Up(base_c * 2, base_c,bilinear, use_attn=False)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # print("x:",x.shape)
        x1 = self.in_conv(x)
        # print("x1:", x1.shape)
        x2 = self.down1(x1)
        # print("x2:", x2.shape)
        x3 = self.down2(x2)
        # print("x3:", x3.shape)
        x4 = self.down3(x3)
        # print("x4:", x4.shape)
        x5 = self.down4(x4)
        # print("x5:", x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)


        return {"out": logits}


