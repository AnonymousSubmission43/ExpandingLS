import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
# from utils import MeanShift
from models.stylegan2.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

import cv2
import numpy as np

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

##_2dfeat
class NoiseInjection_2dfeat(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))
        # self.weight2 = nn.Parameter(torch.zeros(1))

    # when noise is None, add norm noise
    def forward(self, image, noise=None):
        if noise is None:
            # batch, _, height, width = image.shape
            # noise = image.new_empty(batch, 1, height, width).normal_()

            ## return image + self.weight2 * noise
            # return image + self.weight * noise
            return image
        else:
            return self.weight * image * noise
            # return (1 - self.weight) * image + self.weight * noise

## ori
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=False,
            blur_kernel=[1, 3, 3, 1],
            demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.has_upsample = upsample
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.has_upsample:
                skip = self.upsample(skip)

            out = out + skip

        return out

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class LTE(torch.nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1):
        super(LTE, self).__init__()

        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.slice1(x)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        x = self.slice3(x)
        x_lv3 = x
        return x_lv1, x_lv2, x_lv3

class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, queue, key, value):
        print("---->")
        print(queue.size()) # [N, 64, 16, 16]
        print(key.size()) # [N, 64, 16, 16]
        print(value.size()) # [N, 64, 16, 16]
        ### search
        queue_unfold  = F.unfold(queue, kernel_size=(5, 5), padding=1, stride=2) # torch.Size([4, 576, 256])
        key_unfold = F.unfold(key, kernel_size=(5, 5), padding=1, stride=2) # torch.Size([4, 576, 256])
        key_unfold = key_unfold.permute(0, 2, 1)

        key_unfold = F.normalize(key_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        queue_unfold  = F.normalize(queue_unfold, dim=1) # [N, C*k*k, H*W]

        R = torch.bmm(key_unfold, queue_unfold) #[N, Hr*Wr, H*W] #torch.Size([4, 256, 256])
        R_star, R_star_arg = torch.max(R, dim=1) #[N, H*W]
        print(R_star.size())

        ### transfer
        value_unfold = F.unfold(value, kernel_size=(3, 3), padding=1)
        # ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        # ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

        T_unfold = self.bis(value_unfold, 2, R_star_arg)
        # T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_star_arg)
        # T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_star_arg)

        # T = F.fold(T_unfold, output_size=queue.size()[-2:], kernel_size=(5,5), padding=1, stride=2) / (5.*5.)
        # T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        # T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)
        print("T_unfold", T_unfold.size())
        import ipdb; ipdb.set_trace()

        S = R_star.view(R_star.size(0), 1, queue.size(2), queue.size(3))
        print("S", S.size())

        return S, T_unfold # , #T_lv3, T_lv2, T_lv1


class Attention2(nn.Module):
    """ attention Layer"""
    def __init__(self):
        super(Attention,self).__init__()
        # self.chanel_in = in_dim

        self.LTE = LTE(requires_grad=True)
        self.SearchTransfer = SearchTransfer()

    def forward(self, feat_edit=None, feat_ori=None, feat_2d=None):
        """
            inputs :
            queue, key, value
                feat_2d, feat_edit[0], feat_ori[0]
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """

        # edit, _, _ = self.LTE(feat_edit.detach())
        # ori, _, _ = self.LTE(feat_ori.detach())
        #
        # ref, _, _ = self.LTE(feat_2d.detach() )

        # S, T  = self.SearchTransfer(ori, edit, ref)
        S, T  = self.SearchTransfer(feat_ori, feat_edit, feat_2d)

        ### soft-attention
        return S


class ToRGB2(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.has_upsample = upsample
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.has_upsample:
                skip = self.upsample(skip)

            # out = out + skip
            return out, skip

        return out


class Attention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim=3):
        super(Attention,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = 1 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = 1 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = 3 , kernel_size= 1)
        self.value_conv2 = nn.Conv2d(in_channels = in_dim , out_channels = 3 , kernel_size= 1)
        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))
        self.gamma_3 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, feat_edit=None, feat_ori=None, x1=None, x2=None):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        k=16
        m_batchsize, C, width, height = x1.size()
        proj_query  = self.query_conv(feat_edit) # torch.Size([4, 3, 1024, 1024])
        proj_key =  self.key_conv(feat_ori)
        # import ipdb; ipdb.set_trace()

        queue_unfold  = F.unfold(proj_query, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
        key_unfold = F.unfold(proj_key, kernel_size=(k, k), padding=0, stride=k).permute(0,2,1) # B X C x (*W*H)
        key_unfold = F.normalize(key_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        queue_unfold  = F.normalize(queue_unfold, dim=1) # [N, C*k*k, H*W]

        energy = torch.bmm(key_unfold, queue_unfold) #[N, Hr*Wr, H*W] #torch.Size([4, 4096, 4096])

        ##
        energy_star, energy_star_arg = torch.max(energy, dim=1)
        energy_star2, energy_star_arg2 = torch.min(energy, dim=1)
        # print("---------------->")
        # print("energy", energy.size(), energy.min(), energy.max())
        # print("energy_star", energy_star.size(), energy_star.min(), energy_star.max())

        S = energy_star.view(energy_star.size(0), 1, width//k, height//k)
        # print("s", S.size(), S.min(), S.max())

        attention = self.softmax(energy) # B X (N) X (N)
        attention2 = self.softmax(-energy) # B X (N) X (N)


        # value_query = self.value_conv(x1)
        value_query = x1
        value_unfold = F.unfold(value_query, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
        value_unfold_star = self.bis(value_unfold, 2, energy_star_arg)
        # out = torch.bmm(value_unfold, attention)
        out = F.fold(value_unfold_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)
        # print("out", out.size(), out.min(), out.max())

        # value_query2 = self.value_conv2(x2)
        value_query2 = x2
        value_unfold2 = F.unfold(value_query2, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
        value_unfold2_star = self.bis(value_unfold2, 2, energy_star_arg2)
        # out2 = torch.bmm(value_unfold2, attention2)
        out2 = F.fold(value_unfold2_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)

        # print("x1",x1.min(), x1.max())
        # print("x2", x2.min(), x2.max())
        #
        # print("out",out.min(), out.max())
        # print("out2", out2.min(), out2.max())
        # print("att", attention.size(), attention.min(), attention.max())

        # return x1 + self.gamma_3*x2 + self.gamma_1*out + self.gamma_2*out2
        return out + self.gamma_2*out2
        # return out + self.gamma*out2
        # return out + out2
        # proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        #
        # out = torch.bmm(proj_value,attention.permute(0,2,1) )
        # out = out.view(m_batchsize,C,width,height)
        #
        # out = self.gamma*out + x
        # # print("--->gamma:",self.gamma)
        # return out # , attention


class ToRGB_ATT(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], in_dim=3):
        super(ToRGB_ATT,self).__init__()
        self.has_upsample = upsample
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = 16, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = 16, kernel_size= 1)
        # self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = 16, kernel_size= 1)
        # self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = 16, kernel_size= 1)
        # self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = 3, kernel_size= 1)
        # self.value_conv2 = nn.Conv2d(in_channels = in_dim , out_channels = 3 , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma_2 = nn.Parameter(torch.zeros(1))
        # self.gamma_3 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, input, style, skip=None):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.has_upsample:
                skip = self.upsample(skip)
            k=16
            m_batchsize, C, width, height = out.size()
            proj_query  = self.query_conv(out) # torch.Size([4, 3, 1024, 1024])
            proj_key =  self.key_conv(skip)

            queue_unfold  = F.unfold(proj_query, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
            key_unfold = F.unfold(proj_key, kernel_size=(k, k), padding=0, stride=k).permute(0,2,1) # B X C x (*W*H)
            key_unfold = F.normalize(key_unfold, dim=2) # [N, Hr*Wr, C*k*k]
            queue_unfold  = F.normalize(queue_unfold, dim=1) # [N, C*k*k, H*W]

            energy = torch.bmm(key_unfold, queue_unfold) #[N, Hr*Wr, H*W] #torch.Size([4, 4096, 4096])

            energy_star, energy_star_arg = torch.max(energy, dim=1)
            energy_star2, energy_star_arg2 = torch.min(energy, dim=1)

            # value_query = self.value_conv(out)
            value_query = out
            value_unfold = F.unfold(value_query, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
            value_unfold_star = self.bis(value_unfold, 2, energy_star_arg)
            # out = torch.bmm(value_unfold, attention)
            out1 = F.fold(value_unfold_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)
            # print("out", out.size(), out.min(), out.max())

            # value_query2 = self.value_conv2(skip)
            value_query2 = skip
            value_unfold2 = F.unfold(value_query2, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
            value_unfold2_star = self.bis(value_unfold2, 2, energy_star_arg2)
            # out2 = torch.bmm(value_unfold2, attention2)
            out2 = F.fold(value_unfold2_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)

            # out = out1 + self.gamma * skip + self.gamma_2 * out2
            out = out1 + self.gamma * out2
            # out = out1 + self.gamma * skip

            # out = out + skip
        return out


class ToRGB_ATT2(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], in_dim=3):
        super(ToRGB_ATT2,self).__init__()
        self.has_upsample = upsample
        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = 1, kernel_size= 3, padding=1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = 1, kernel_size= 3, padding=1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = 3, kernel_size= 1)
        self.value_conv2 = nn.Conv2d(in_channels = in_dim , out_channels = 3 , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma_2 = nn.Parameter(torch.zeros(1))
        # self.gamma_3 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=0) #

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)


    def forward(self, input, style, skip=None, attention=None):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.has_upsample:
                skip = self.upsample(skip)
            k=4
            m_batchsize, C, width, height = out.size()
            if width > k:
                # out 2d infor
                # skip 1d infor
                proj_query  = self.query_conv(attention[0].detach()) # torch.Size([4, 3, 1024, 1024])
                proj_key =  self.query_conv(attention[1].detach())
                # proj_key =  self.key_conv(attention[1].detach())
                # print("proj_query",proj_query.size())
                queue_unfold  = F.unfold(proj_query, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
                key_unfold = F.unfold(proj_key, kernel_size=(k, k), padding=0, stride=k) # B X C x (*W*H)
                queue_unfold  = F.normalize(queue_unfold, dim=1) # [N, C*k*k, H*W]
                key_unfold = F.normalize(key_unfold, dim=1) # [N, C*k*k, H*W]
                # print("queue_unfold",queue_unfold.size())

                # energy = torch.bmm(key_unfold, queue_unfold) #[N, Hr*Wr, H*W] #torch.Size([4, 4096, 4096])
                energy = key_unfold * queue_unfold
                # print("--->")
                # print(energy.size(), energy.min(), energy.max(), energy.mean())
                energy_star = F.normalize(energy, dim=(1, 2)) * k
                # energy_star  = self.softmax(energy)
                # energy_star, energy_star_arg = torch.max(energy, dim=1, keepdim=True)
                # print(energy_star.size(), energy_star.min(), energy_star.max(), energy_star.mean())
                # energy = torch.sum(key_unfold * queue_unfold, dim=1, keepdim=True)

                # energy = torch.sum(key_unfold * queue_unfold, dim=1, keepdim=True)
                # energy = F.mse_loss(key_unfold * queue_unfold, 0)
                # print("--> bef0 softmax ",energy.min(), energy.max(), energy.mean())
                # energy  = F.normalize(energy, dim=1)
                # with torch.no_grad():
                #     print(torch.norm(energyt, p=2, dim=1, keepdim=True).size())
                #     energy = energyt.div_(torch.norm(energyt, p=2, dim=1, keepdim=True))
                # print("--> bef softmax ",energy.size(), energy.min(), energy.max(), energy.mean())
                # plot_feat_map(energy.unsqueeze(1).detach(), save_path=f'../experiments/pdsp_wClassfier/temp/energy1.jpg')

                # attention_map = (1 + energy) / 2.0 #self.softmax(-energy)
                # attention_map2 = 1 - (1 + energy) / 2.0 #self.softmax(energy)
                # energy_star = (1 + energy_star) / 2.0
                attention_map = F.fold(energy_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)
                # print(attention_map.size())
                # plot_feat_map(attention_map.detach(), save_path=f'../experiments/pdsp_wClassfier/temp/energy3.jpg')
                # plot_feat_map(attention_map.unsqueeze(1).detach(), save_path=f'../experiments/pdsp_wClassfier/temp/energy3.jpg')
                # plot_feat_map(attention_map2.unsqueeze(1).detach(), save_path=f'../experiments/pdsp_wClassfier/temp/energy5.jpg')
                # print("aft softmax1 ",attention_map.size(),attention_map.min(), attention_map.max(), attention_map.mean())
                # print("aft softmax2 ",attention_map2.min(), attention_map2.max(), attention_map2.mean())
                #
                # # value_query = self.value_conv(out)
                # value_query = out
                # value_unfold = F.unfold(value_query, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
                # # value_unfold_star = self.bis(value_unfold, 2, energy_star_arg)
                # value_unfold_star = value_unfold * attention_map
                # # value_unfold_star = torch.bmm(value_unfold, attention_map)
                # out1 = F.fold(value_unfold_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)
                # # print("out", out.size(), out.min(), out.max())
                #
                # # value_query2 = self.value_conv2(skip)
                # value_query2 = skip
                # value_unfold2 = F.unfold(value_query2, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
                # # value_unfold2_star = self.bis(value_unfold2, 2, energy_star_arg2)
                # value_unfold2_star = value_unfold2 * attention_map2
                # # value_unfold2_star = torch.bmm(value_unfold2, attention_map2)
                # out2 = F.fold(value_unfold2_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)


                # energy_star, energy_star_arg = torch.max(energy, dim=1)
                # energy_star2, energy_star_arg2 = torch.min(energy, dim=1)
                #
                # value_query = self.value_conv(out)
                # # value_query = out
                # value_unfold = F.unfold(value_query, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
                # value_unfold_star = self.bis(value_unfold, 2, energy_star_arg)
                # # out = torch.bmm(value_unfold, attention)
                # out1 = F.fold(value_unfold_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)
                # # print("out", out.size(), out.min(), out.max())
                #
                # value_query2 = self.value_conv2(skip)
                # # value_query2 = skip
                # value_unfold2 = F.unfold(value_query2, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
                # value_unfold2_star = self.bis(value_unfold2, 2, energy_star_arg2)
                # # out2 = torch.bmm(value_unfold2, attention2)
                # out2 = F.fold(value_unfold2_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)

                # out = out1 + self.gamma * skip + self.gamma_2 * out2
                # out = torch.bmm(out, out1) + self.gamma * torch.bmm(skip, out2)
                out = out * ( attention_map) + skip * (1 - attention_map)
                # out = out * (1 -  attention_map) + skip * (attention_map)
                # out = out * attention_map + self.gamma * skip * (1-attention_map)
                # out =  out1 + self.gamma * out2
                # out = out1 + self.gamma * skip

                # out = out + skip
            else:
                out = out + self.gamma * skip
        return out



    def forward1(self, input, style, skip=None, attention=None):
        """
        set self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = 1, kernel_size= 3, padding=1)
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            if self.has_upsample:
                skip = self.upsample(skip)
            k=16
            m_batchsize, C, width, height = out.size()
            if width > k:
                # out 2d infor
                # skip 1d infor
                proj_query  = self.query_conv(attention[0].detach()) # torch.Size([4, 3, 1024, 1024])
                proj_key =  self.query_conv(attention[1].detach())
                # proj_key =  self.key_conv(attention[1].detach())
                # print("proj_query",proj_query.size())
                queue_unfold  = F.unfold(proj_query, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
                key_unfold = F.unfold(proj_key, kernel_size=(k, k), padding=0, stride=k) # B X C x (*W*H)
                queue_unfold  = F.normalize(queue_unfold, dim=1) # [N, C*k*k, H*W]
                key_unfold = F.normalize(key_unfold, dim=1) # [N, C*k*k, H*W]
                # print("queue_unfold",queue_unfold.size())

                # energy = torch.bmm(key_unfold, queue_unfold) #[N, Hr*Wr, H*W] #torch.Size([4, 4096, 4096])
                energy = key_unfold * queue_unfold
                # energy = torch.sum(key_unfold * queue_unfold, dim=1, keepdim=True)

                # energy = torch.sum(key_unfold * queue_unfold, dim=1, keepdim=True)
                # energy = F.mse_loss(key_unfold * queue_unfold, 0)
                # print("--> bef0 softmax ",energy.min(), energy.max(), energy.mean())
                # energy  = F.normalize(energy, dim=1)
                # with torch.no_grad():
                #     print(torch.norm(energyt, p=2, dim=1, keepdim=True).size())
                #     energy = energyt.div_(torch.norm(energyt, p=2, dim=1, keepdim=True))
                # print("--> bef softmax ",energy.size(), energy.min(), energy.max(), energy.mean())
                # plot_feat_map(energy.unsqueeze(1).detach(), save_path=f'../experiments/pdsp_wClassfier/temp/energy1.jpg')

                # attention_map = (1 + energy) / 2.0 #self.softmax(-energy)
                # attention_map2 = 1 - (1 + energy) / 2.0 #self.softmax(energy)
                energy = (1 + energy) / 2.0
                attention_map = F.fold(energy, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)
                # print(attention_map.size())
                # plot_feat_map(attention_map.detach(), save_path=f'../experiments/pdsp_wClassfier/temp/energy3.jpg')
                # plot_feat_map(attention_map.unsqueeze(1).detach(), save_path=f'../experiments/pdsp_wClassfier/temp/energy3.jpg')
                # plot_feat_map(attention_map2.unsqueeze(1).detach(), save_path=f'../experiments/pdsp_wClassfier/temp/energy5.jpg')
                # print("aft softmax1 ",attention_map.size(),attention_map.min(), attention_map.max(), attention_map.mean())
                # print("aft softmax2 ",attention_map2.min(), attention_map2.max(), attention_map2.mean())
                #
                # # value_query = self.value_conv(out)
                # value_query = out
                # value_unfold = F.unfold(value_query, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
                # # value_unfold_star = self.bis(value_unfold, 2, energy_star_arg)
                # value_unfold_star = value_unfold * attention_map
                # # value_unfold_star = torch.bmm(value_unfold, attention_map)
                # out1 = F.fold(value_unfold_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)
                # # print("out", out.size(), out.min(), out.max())
                #
                # # value_query2 = self.value_conv2(skip)
                # value_query2 = skip
                # value_unfold2 = F.unfold(value_query2, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
                # # value_unfold2_star = self.bis(value_unfold2, 2, energy_star_arg2)
                # value_unfold2_star = value_unfold2 * attention_map2
                # # value_unfold2_star = torch.bmm(value_unfold2, attention_map2)
                # out2 = F.fold(value_unfold2_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)


                # energy_star, energy_star_arg = torch.max(energy, dim=1)
                # energy_star2, energy_star_arg2 = torch.min(energy, dim=1)
                #
                # value_query = self.value_conv(out)
                # # value_query = out
                # value_unfold = F.unfold(value_query, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
                # value_unfold_star = self.bis(value_unfold, 2, energy_star_arg)
                # # out = torch.bmm(value_unfold, attention)
                # out1 = F.fold(value_unfold_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)
                # # print("out", out.size(), out.min(), out.max())
                #
                # value_query2 = self.value_conv2(skip)
                # # value_query2 = skip
                # value_unfold2 = F.unfold(value_query2, kernel_size=(k, k), padding=0, stride=k) # B X CX(N)
                # value_unfold2_star = self.bis(value_unfold2, 2, energy_star_arg2)
                # # out2 = torch.bmm(value_unfold2, attention2)
                # out2 = F.fold(value_unfold2_star, output_size=(width, height), kernel_size=(k,k), padding=0, stride=k)

                # out = out1 + self.gamma * skip + self.gamma_2 * out2
                # out = torch.bmm(out, out1) + self.gamma * torch.bmm(skip, out2)
                out = out * ( attention_map) + skip * (1 - attention_map)
                # out = out * attention_map + self.gamma * skip * (1-attention_map)
                # out =  out1 + self.gamma * out2
                # out = out1 + self.gamma * skip

                # out = out + skip
            else:
                out = out + self.gamma * skip
        return out


## ada_in 2d features using new code, and add to to_rgb part, att #
class Generator_adain_2d_using_new_code_on_toRGB_ATT2(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        self.fuse = nn.ModuleList()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            self.fuse.append(ToRGB_ATT2(out_channel, style_dim, upsample=False))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

        # for j in [4, 8, 16, 32, 64, 128, 256, 512, 1024]: ## 15
            # self.fuse.append(nn.Conv2d(self.channels[j], 3, 1))

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
            self,
            styles,
            return_latents=False,
            return_features=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
            alpha=1.0,
            struct=None,
            edit_2d=None,
            attention=None
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        if return_features:
            mid_feat = []

        if struct is not None:
            out = self.input(latent)
            out = self.conv1(out, latent[:, 0], noise=noise[0])

            skip = self.to_rgb1(out, latent[:, 1])
            # idx = torch.randint(100000, (1,))
            # plot_feat_map(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_0_skip.jpg')
            # plot_feat_map(self.fuse[0](alpha * struct[0]), save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_0_struct.jpg')
            skip = skip + self.fuse[0](alpha * struct[0], edit_2d[:, 0])
            # plot_feat_map(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_0_out.jpg')

            i = 1
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                    self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            ):
                out = conv1(out, latent[:, i], noise=noise1)
                out = conv2(out, latent[:, i + 1], noise=noise2)
                if i <= 18:
                    skip = to_rgb(out, latent[:, i + 2], skip)
                    # plot_feat_map(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_skip.jpg')
                    # plot_feat_map(self.fuse[(i+1)//2](alpha * struct[i+2]), save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_struct.jpg')

                    # skip = skip + self.fuse[i//2](alpha * struct[i+2], edit_2d[:, i//2 + 1])
                    skip = self.fuse[i//2](alpha * struct[i+2], edit_2d[:, i//2 + 1], skip,
                                           [attention[0][i//2+1], attention[1][i//2+1]])
                    # plot_feat_map(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_out.jpg')
                else:
                    skip = to_rgb(out, latent[:, i + 2], skip)
                    # plot_feat_map(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_skipout.jpg')

                i += 2
            ## content
            # out = self.input(latent)
            # out = self.conv1(out + alpha * struct[0], latent[:, 0], noise=noise[0])
            #
            # skip = self.to_rgb1(out, latent[:, 1])
            #
            # i = 1
            # for conv1, conv2, noise1, noise2, to_rgb in zip(
            #         self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            # ):
            #     out = conv1(out + alpha * struct[i], latent[:, i], noise=noise1)
            #     out = conv2(out + alpha * struct[i+1], latent[:, i + 1], noise=noise2)
            #     skip = to_rgb(out + alpha * struct[i+2], latent[:, i + 2], skip)
            #
            #     i += 2
        else:
            out = self.input(latent)

            # out = self.input(latent) + struct if struct is not None else self.input(latent)
            out = self.conv1(out, latent[:, 0], noise=noise[0])

            skip = self.to_rgb1(out, latent[:, 1])
            if return_features:
                mid_feat.append(skip)

            i = 1
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                    self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            ):
                out = conv1(out, latent[:, i], noise=noise1)
                # if i == 3 and struct is not None:
                #     out = out + struct

                out = conv2(out, latent[:, i + 1], noise=noise2)
                skip = to_rgb(out, latent[:, i + 2], skip)
                if return_features:
                    mid_feat.append(skip)

                i += 2

        image = skip

        if return_latents:
            return image, latent
        elif return_features:
            return image, mid_feat
        else:
            return image, None


## ada_in 2d features using new code, and add to to_rgb part, att
class Generator(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        self.fuse = nn.ModuleList()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            self.fuse.append(ToRGB_ATT(out_channel, style_dim, upsample=False))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

        # for j in [4, 8, 16, 32, 64, 128, 256, 512, 1024]: ## 15
            # self.fuse.append(nn.Conv2d(self.channels[j], 3, 1))

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
            self,
            styles,
            return_latents=False,
            return_features=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
            alpha=1.0,
            struct=None,
            edit_2d=None
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        if struct is not None:
            out = self.input(latent)
            out = self.conv1(out, latent[:, 0], noise=noise[0])

            skip = self.to_rgb1(out, latent[:, 1])
            # idx = torch.randint(100000, (1,))
            # plot_feat_map(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_0_skip.jpg')
            # plot_feat_map(self.fuse[0](alpha * struct[0]), save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_0_struct.jpg')
            skip = skip + self.fuse[0](alpha * struct[0], edit_2d[:, 0])
            # plot_feat_map(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_0_out.jpg')

            i = 1
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                    self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            ):
                out = conv1(out, latent[:, i], noise=noise1)
                out = conv2(out, latent[:, i + 1], noise=noise2)
                if i <= 18:
                    skip = to_rgb(out, latent[:, i + 2], skip)
                    # plot_feat_map(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_skip.jpg')
                    # plot_feat_map(self.fuse[(i+1)//2](alpha * struct[i+2]), save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_struct.jpg')

                    skip = skip + self.fuse[i//2](alpha * struct[i+2], edit_2d[:, i//2 + 1])
                    # plot_feat_map(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_out.jpg')
                else:
                    skip = to_rgb(out, latent[:, i + 2], skip)
                    # plot_feat_map(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_skipout.jpg')

                i += 2
            ## content
            # out = self.input(latent)
            # out = self.conv1(out + alpha * struct[0], latent[:, 0], noise=noise[0])
            #
            # skip = self.to_rgb1(out, latent[:, 1])
            #
            # i = 1
            # for conv1, conv2, noise1, noise2, to_rgb in zip(
            #         self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            # ):
            #     out = conv1(out + alpha * struct[i], latent[:, i], noise=noise1)
            #     out = conv2(out + alpha * struct[i+1], latent[:, i + 1], noise=noise2)
            #     skip = to_rgb(out + alpha * struct[i+2], latent[:, i + 2], skip)
            #
            #     i += 2
        else:
            out = self.input(latent)

            # out = self.input(latent) + struct if struct is not None else self.input(latent)
            out = self.conv1(out, latent[:, 0], noise=noise[0])

            skip = self.to_rgb1(out, latent[:, 1])

            i = 1
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                    self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            ):
                out = conv1(out, latent[:, i], noise=noise1)
                # if i == 3 and struct is not None:
                #     out = out + struct

                out = conv2(out, latent[:, i + 1], noise=noise2)
                skip = to_rgb(out, latent[:, i + 2], skip)

                i += 2

        image = skip

        if return_latents:
            return image, latent
        elif return_features:
            return image, out
        else:
            return image, None


## without 2d feat #
class Generator_ori(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
            self,
            styles,
            return_latents=False,
            return_features=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
            input_is_latent=False,
            noise=None,
            randomize_noise=True,
            struct=None
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        # out = struct if struct is not None else self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])


        # idx = torch.randint(100000, (1,))
        # plot_feat_map(out, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_0_conv1.jpg')
        # plot_feat_map(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_0_skip.jpg')
        # plot_feat_map_sum(out, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_0_conv1_sum.jpg')
        # plot_feat_map_sum(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_0_skip_sum.jpg')

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            # if i == 3 and struct is not None:
            #     out = out + struct
            # plot_feat_map(out, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_conv1.jpg')
            # plot_feat_map_sum(out, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_conv1_sum.jpg')

            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            # plot_feat_map(out, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_conv2.jpg')
            # plot_feat_map(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_skip.jpg')
            # plot_feat_map_sum(out, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_conv2_sum.jpg')
            # plot_feat_map_sum(skip, save_path=f'../experiments/pdsp_wClassfier/temp/{idx}_{i}_skip_sum.jpg')

            i += 2

        image = skip

        if return_latents:
            return image, latent
        elif return_features:
            return image, out
        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
