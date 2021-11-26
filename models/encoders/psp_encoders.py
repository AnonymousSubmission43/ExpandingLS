import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE,  _upsample_add
from models.stylegan2.model import EqualLinear
import random
import math
from enum import Enum
from models.stylegan2.op import FusedLeakyReLU

class ProgressiveStage(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Delta14Training = 14
    Delta15Training = 15
    Delta16Training = 16
    Delta17Training = 17
    Inference = 18

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualContentBlock(Module):
    def __init__(self, in_c, out_c, num_pools=0, mid_c=64, num_layer=4, upsample=0):
        super(GradualContentBlock, self).__init__()
        self.out_c = out_c
        modules = []

        for i in range(num_layer):
            if num_pools > 0:
                num_pools -= 1
                strid = 2
            else:
                strid = 1
            if i > 0:
                in_c = mid_c
            if i == num_layer - 1:
                mid_c = out_c
            modules += [
                Conv2d(in_c, mid_c, kernel_size=3, stride=strid, padding=1),
                # BatchNorm2d(mid_c),
                FusedLeakyReLU(mid_c)
            ]

        for i in range(upsample//2):
            modules.append(Conv2d(mid_c, 4 * mid_c, kernel_size=3, padding=1))
            # modules.append(BatchNorm2d(4 * mid_c))
            modules.append(nn.PixelShuffle(2))

        self.convs = nn.Sequential(*modules)
        # self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        # x = x.view(-1, self.out_c)
        # x = self.linear(x)
        return x

class GradualStyleContentEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None, binary_mask_flag=False):
        super(GradualStyleContentEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        self.input_binary_mask_flag = opts.input_binary_mask_flag
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.contents = nn.ModuleList()
        for i in range(self.style_count):
            if i < 2: #16 -> 4
                content = GradualContentBlock(512, 512, 2)
            elif i < 4: # 16 -> 8
                content = GradualContentBlock(512, 512, 1)
            elif i < 10:
                content = GradualContentBlock(512, 512, 0)
            elif i < 12:
                content = GradualContentBlock(512, 256, 0)
            # else:
            #     content = GradualContentBlock(512, 128, 0)
            elif i < 14:
                content = GradualContentBlock(512, 128, 0)
            elif i < 16:
                content = GradualContentBlock(512, 64, 0, upsample=2)
            else:
                content = GradualContentBlock(512, 32, 0, upsample=4)
            self.contents.append(content)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def apply_randonm_binary_mask(self, f2d):
        for i in range(len(f2d)):
            x0 = random.randint(0, f2d.size(-2) - 2)
            x1 = random.randint(x0 + 1, f2d.size(-2))
            y0 = random.randint(0, f2d.size(-1) - 2)
            y1 = random.randint(y0 + 1, f2d.size(-1))
            f2d[i, :, x0:x1, y0:y1] = 0.0
        return f2d

    def apply_consistent_binary_mask(self, f2d, axis):
        for i in range(len(f2d)):
            [x0, x1, y0, y1] = axis[i]
            f2d[i, :, x0:x1, y0:y1] = 0.0
            # f2d[i, :, x0:x1, y0:y1] = f2d[i-1, :, x0:x1, y0:y1].detach()
        return f2d

    def get_binary_map_axis(self, dx, dy, batch_size):
        axis = np.empty((batch_size, 4), dtype=np.int)
        for i in range(batch_size):
            x0 = random.randint(0, dx - 2)
            x1 = random.randint(x0 + 2, dx)
            y0 = random.randint(0, dy - 2)
            y1 = random.randint(y0 + 2, dy)
            axis[i] = np.array([x0, x1, y0, y1])
        return axis


    def forward(self, x, no_content_flag=False):
        x = self.input_layer(x)

        latents = []
        latents2D = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c0 = x
            elif i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        if self.input_binary_mask_flag:
            print("has mask???")
            if no_content_flag:
                axis = np.repeat([[0, 4, 0, 4]], x.size(0), axis=1)
            else:
                axis = self.get_binary_map_axis(4, 4, x.size(0))

        p0 = self._upsample_add(p1.detach(), self.latlayer3(c0.detach()))
        c00 = F.interpolate(c0.detach(), size=(2*c0.size()[-2], 2*c0.size()[-1]), mode='bilinear', align_corners=True)
        p00 = self._upsample_add(p0, self.latlayer4(c00))

        for k in range(self.style_count):
            if k < 6: # 4,8,16
                latent2D = self.contents[k](c3.detach())
            elif k < 8: #32
                latent2D = self.contents[k](p2.detach())
            elif k < 10: #64
                latent2D = self.contents[k](p1.detach())
            elif k < 12:
                latent2D = self.contents[k](p0.detach())
            else:
                latent2D = self.contents[k](p00)

            if self.input_binary_mask_flag:
                latent2D = self.apply_consistent_binary_mask(latent2D, (latent2D.size(-1) // 4 + 1)*axis)
            latents2D.append(latent2D)

        out = torch.stack(latents, dim=1)
        return out, latents2D

class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.n_styles = opts.n_styles
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(BatchNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = EqualLinear(512, 512 * self.n_styles, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, self.n_styles, 512)
        return x

class Encoder4Editing(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(Encoder4Editing, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        # self.style_2d = Sequential(
        #                            unit_module(512, 512, 2),
        #                            unit_module(512, 512, 1),
        #                            unit_module(512, 512, 1),
        #                            unit_module(512, 512, 2),
        #                            unit_module(512, 512, 1),
        #                            unit_module(512, 512, 1),
        #                            )

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        # # 2D features
        # feat_2d = self.style_2d(c3)
        # return w, feat_2d
        return w


class Encoder4ContentEditing(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(Encoder4ContentEditing, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        # self.style_2d = Sequential(
        #                            unit_module(512, 512, 2),
        #                            unit_module(512, 512, 1),
        #                            unit_module(512, 512, 1),
        #                            unit_module(512, 512, 2),
        #                            unit_module(512, 512, 1),
        #                            unit_module(512, 512, 1),
        #                            )

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)


        self.contents = nn.ModuleList()
        for i in range(self.style_count):
            if i < 2: #16 -> 4
                content = GradualContentBlock(512, 512, 2)
            elif i < 4: # 16 -> 8
                content = GradualContentBlock(512, 512, 1)
            elif i < 10:
                content = GradualContentBlock(512, 512, 0)
            elif i < 12:
                content = GradualContentBlock(512, 256, 0)
            # else:
            #     content = GradualContentBlock(512, 128, 0)
            elif i < 14:
                content = GradualContentBlock(512, 128, 0)
            elif i < 16:
                content = GradualContentBlock(512, 64, 0, upsample=2)
            else:
                content = GradualContentBlock(512, 32, 0, upsample=4)
            self.contents.append(content)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c0 = x
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x


        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        # 2D features
        latents2D = []
        p0 = _upsample_add(p1.detach(), self.latlayer3(c0.detach()))
        c00 = F.interpolate(c0.detach(), size=(2*c0.size()[-2], 2*c0.size()[-1]), mode='bilinear', align_corners=True)
        p00 = _upsample_add(p0, self.latlayer4(c00))

        for k in range(self.style_count):
            if k < 6: # 4,8,16
                latent2D = self.contents[k](c3.detach())
            elif k < 8: #32
                latent2D = self.contents[k](p2.detach())
            elif k < 10: #64
                latent2D = self.contents[k](p1.detach())
            elif k < 12:
                latent2D = self.contents[k](p0.detach())
            else:
                latent2D = self.contents[k](p00)


            latents2D.append(latent2D)

        # feat_2d = self.style_2d(c3)
        return w, latents2D
        # return w


class Branch2(Module):
    def __init__(self):
        super(Branch2, self).__init__()
        self.style2 = GradualStyleBlock(512, 512, 16)

    def forward(self, c3):
        # Infer a second branch W
        w0 = self.style2(c3)

class Encoder4Editing2Branch(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(Encoder4Editing2Branch, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        return w, c3


class GradualStyleEncoder2Branch(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder2Branch, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.styles_2nd = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
                style_2nd = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
                style_2nd = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
                style_2nd = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
            self.styles_2nd.append(style_2nd)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        latents_2nd = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))
            latents_2nd.append(self.styles_2nd[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))
            latents_2nd.append(self.styles_2nd[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))
            latents_2nd.append(self.styles_2nd[j](p1))

        out = torch.stack(latents, dim=1)
        out_2nd = torch.stack(latents_2nd, dim=1)
        return out, out_2nd


class EncodertoW2Branch(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(EncodertoW2Branch, self).__init__()
        print('Using EncodertoW2Branch')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        self.linear_2 = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x_1 = self.linear(x)
        x_2 = self.linear_2(x)
        return x_1, x_2

