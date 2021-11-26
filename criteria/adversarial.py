from utils import train_utils
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# def make_optimizer(target):
#     params = list(target.direction_encoder.parameters())

#     if self.opts.optim_name == 'adam':
#         optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate_direction)
#     else:
#         optimizer = Ranger(params, lr=self.opts.learning_rate_direction)
#     return optimizer

class Discriminator(nn.Module):
    '''
        output is not normalized
    '''
    def __init__(self, img_size=256):
        super(Discriminator, self).__init__()

        in_channels = 3
        out_channels = 64
        depth = 7
        act = nn.LeakyReLU(True)

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    3,
                    padding=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(_out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        m_features = [_block(in_channels, out_channels)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(_block(in_channels, out_channels, stride=stride))

        patch_size = img_size // (2**((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size**2, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        ]

        self.features = nn.Sequential(*m_features)
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output

class Adversarial(nn.Module):
    def __init__(self, gan_type='WGAN_GP'):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = 1
        self.dis = Discriminator(img_size=256)

        optim_dict = {
            'optimizer': 'ADAM',
            'betas': (0, 0.9),
            'epsilon': 1e-8,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            # 'decay': '10000000-20',
            'gamma': 0.5
        }
        optim_args = SimpleNamespace(**optim_dict)

        self.optimizer = train_utils.make_optimizer(optim_args, self.dis)

    def forward(self, outputs, targets):
        fake = outputs
        real = targets
        # updating discriminator...
        self.loss = 0
        fake_detach = fake.detach()     # do not backpropagate through G
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            # d: B x 1 tensor
            d_fake = self.dis(fake_detach)
            d_real = self.dis(real)
            retain_graph = False
            if self.gan_type == 'GAN':
                loss_d = self.bce(d_real, d_fake)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    # epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    epsilon = torch.rand(fake.shape[0], 1, 1, 1)
                    epsilon = epsilon.expand_as(real)
                    epsilon = epsilon.cuda()

                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.dis(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            # Discriminator update
            self.loss += loss_d.item()
            loss_d.backward(retain_graph=retain_graph)
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.dis.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        # updating generator...
        d_fake_bp = self.dis(fake)      # for backpropagation, use fake as it is
        if self.gan_type == 'GAN':
            label_real = torch.ones_like(d_fake_bp)
            loss_g = F.binary_cross_entropy_with_logits(d_fake_bp, label_real)
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_bp.mean()

        # Generator loss
        return loss_g
    
    def state_dict(self, *args, **kwargs):
        state_discriminator = self.dis.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)

    def bce(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        bce_real = F.binary_cross_entropy_with_logits(real, label_real)
        bce_fake = F.binary_cross_entropy_with_logits(fake, label_fake)
        bce_loss = bce_real + bce_fake
        return bce_loss
               
# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
