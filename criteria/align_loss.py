import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from configs.paths_config import model_paths

from torch.autograd import Variable
import numpy as np
from math import exp


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


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def compute_ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    # return 1+ssim_map
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class ssim_loss(nn.Module):
    def __init__(self):
        super(ssim_loss, self).__init__()
        self.window_size = 31
        self.size_average = True
        self.channel = 1
        self.window = create_window(self.window_size, self.channel)


    def forward(self, img1, img2):
        img1 = ((img1 + 1) / 2)
        img1[img1 < 0] = 0
        img1[img1 > 1] = 1
        img2 = ((img2 + 1) / 2)
        img2[img2 < 0] = 0
        img2[img2 > 1] = 1

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - compute_ssim(img1, img2, window, self.window_size, channel, self.size_average)


class AlignLoss(nn.Module):

    def __init__(self):
        super(AlignLoss, self).__init__()
        # vgg_pretrained_features = models.vgg19(pretrained=True).features
        #
        # self.slice1 = torch.nn.Sequential()
        # self.slice2 = torch.nn.Sequential()
        # self.slice3 = torch.nn.Sequential()
        #
        # for x in range(2):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(2, 7):
        #     self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # for x in range(7, 12):
        #     self.slice3.add_module(str(x), vgg_pretrained_features[x])
        #
        # rgb_range = 1
        # vgg_mean = (0.485, 0.456, 0.406)
        # vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        # self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        #
        # self.ssim_loss = ssim_loss()
        # # self.model = self.__load_model()
        # # self.model.cuda()
        # # self.model.eval()

    @staticmethod
    def __load_model():
        import torchvision.models as models
        model = models.__dict__["resnet50"]()
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            print(name)
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        # remove output layer
        model = nn.Sequential(*list(model.children())[:-1]).cuda()
        return model

    def extract_feats(self, x):
        x = ((x + 1) / 2)
        x[x < 0] = 0
        x[x > 1] = 1

        x = self.sub_mean(x)


        x = self.slice1(x)
        # x_lv1 = nn.functional.normalize(x, dim=1)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        x = self.slice3(x)
        x_lv3 = x

        return [x_lv1]
        # return [x_lv1, x_lv2, x_lv3]

        # x = F.interpolate(x, size=224)
        # x_feats = self.model(x)
        # x_feats = nn.functional.normalize(x_feats, dim=1)
        # x_feats = x_feats.squeeze()
        # return x_feats

    def forward(self, y_hat_edit, x, y_hat, rec2):
        n_samples = x.shape[0]
        ## imgage-based
        diff = (y_hat_edit - x.detach()) - (y_hat.detach() - rec2.detach())
        loss_align = torch.mean(diff.norm(1, dim=(1,2,3)))

        # ## ssim-based
        # loss_align = self.ssim_loss(y_hat_edit, x.detach()) - self.ssim_loss(y_hat.detach(), rec2.detach())

        # ## feat-based
        # align1_y_hat_edit = self.extract_feats(y_hat_edit)
        # align1_x = self.extract_feats(x.detach())
        # align2_y_hat = self.extract_feats(y_hat.detach())
        # align2_rec2= self.extract_feats(rec2.detach())
        #
        # loss_align = 0
        # for i in range(len(align1_y_hat_edit)):
        #     loss_align_1 = (align1_y_hat_edit[i] - align1_x[i])
        #     loss_align_1 = loss_align_1.norm(2, dim=(1,2,3))
        #     loss_align_2 = (align2_y_hat[i] - align2_rec2[i])
        #     loss_align_2 = loss_align_2.norm(2, dim=(1,2,3))
        #     # print((loss_align_1 - loss_align_2))
        #     loss_align += torch.mean((loss_align_1 - loss_align_2))
        #     # loss_align += torch.mean((loss_align_1 - loss_align_2).norm(2, dim=(1,2,3)))


        # print("loss_align",loss_align / n_samples)


        return loss_align / n_samples


        # n_samples = x.shape[0]
        # x_feats = self.extract_feats(x)
        # y_feats = self.extract_feats(y)
        # y_hat_feats = self.extract_feats(y_hat)
        # y_feats = y_feats.detach()
        # loss = 0
        # sim_improvement = 0
        # sim_logs = []
        # count = 0
        # for i in range(n_samples):
        #     diff_target = y_hat_feats[i].dot(y_feats[i])
        #     diff_input = y_hat_feats[i].dot(x_feats[i])
        #     diff_views = y_feats[i].dot(x_feats[i])
        #     sim_logs.append({'diff_target': float(diff_target),
        #                      'diff_input': float(diff_input),
        #                      'diff_views': float(diff_views)})
        #     loss += 1 - diff_target
        #     sim_diff = float(diff_target) - float(diff_views)
        #     sim_improvement += sim_diff
        #     count += 1
        #
        # return loss / count
