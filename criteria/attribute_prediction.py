import os.path

# import torch
# from torch import nn
# from configs.paths_config import model_paths
# from models.encoders.model_irse import Backbone

import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F

ATTRIBUTES = ['5_o_Clock_Shadow', 'Arched_Eyebrows',
        'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
        'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
        'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
        'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young']

def get_resnet():
    net = resnet18()
    modified_net = nn.Sequential(*list(net.children())[:-1])  # fetch all of the layers before the last fc.
    return modified_net


class ClassifyModel(nn.Module):
    def __init__(self, n_class=2):
        super(ClassifyModel, self).__init__()
        self.backbone = get_resnet()
        self.extra_layer = nn.Linear(512, n_class)

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        out = self.extra_layer(out)
        return out


class AttrPredLoss_5(nn.Module):
    def __init__(self, pretrained_model_path=None, dual_direction=False):
        super(AttrPredLoss_5, self).__init__()
        model_path_smile ="IALS_pretrain/classifier/smiling/weight.pkl"
        self.classifier_smile = ClassifyModel().eval()
        self.classifier_smile.load_state_dict(torch.load(os.path.join(
            pretrained_model_path, model_path_smile), map_location=torch.device('cuda')))

        model_path_eyeglasses ="IALS_pretrain/classifier/eyeglasses/weight.pkl"
        self.classifier_eyeglasses = ClassifyModel().eval()
        self.classifier_eyeglasses.load_state_dict(torch.load(os.path.join(
            pretrained_model_path, model_path_eyeglasses), map_location=torch.device('cuda')))

        model_path_male ="IALS_pretrain/classifier/male/weight.pkl"
        self.classifier_male = ClassifyModel().eval()
        self.classifier_male.load_state_dict(torch.load(os.path.join(
            pretrained_model_path, model_path_male), map_location=torch.device('cuda')))

        model_path_pose ="IALS_pretrain/classifier/pose/weight.pkl"
        self.classifier_pose = ClassifyModel().eval()
        self.classifier_pose.load_state_dict(torch.load(os.path.join(
            pretrained_model_path, model_path_pose), map_location=torch.device('cuda')))

        model_path_young ="IALS_pretrain/classifier/young/weight.pkl"
        self.classifier_young = ClassifyModel().eval()
        self.classifier_young.load_state_dict(torch.load(os.path.join(
            pretrained_model_path, model_path_young), map_location=torch.device('cuda')))

        self.loss_f = nn.CrossEntropyLoss().to('cuda')
        self.dual_direction = dual_direction
        # self.softmax = nn.Softmax(dim=1).to('cuda')
        # self.loss_f = nn.MSELoss().to('cuda')
        print(f'Loading classifier from {model_path_young}')

    def softargmax1d(self, input, beta=100):
        *_, n = input.shape
        input = nn.functional.softmax(beta * input, dim=-1)
        indices = torch.linspace(0, 1, n).to('cuda')
        result = torch.sum((n - 1) * input * indices, dim=-1)
        return result

    def forward(self, y_hat, label):
        y_hat = (y_hat + 1) / 2

        # print(f"gt attributes: {label}")
        # print(f"predict pose: {torch.argmax(self.classifier_pose(y_hat), 1)}")
        # print(f"predict eyeglasses: {torch.argmax(self.classifier_eyeglasses(y_hat), 1)}")
        # print(f"predict young: {torch.argmax(self.classifier_young(y_hat), 1)}")
        # print(f"predict male: {torch.argmax(self.classifier_male(y_hat), 1)}")
        # print(f"predict smile: {torch.argmax(self.classifier_smile(y_hat), 1)}")
        # import ipdb; ipdb.set_trace()

        label = label.long()
        pose = self.classifier_pose(y_hat)
        eye_glasses = self.classifier_eyeglasses(y_hat)
        young = self.classifier_young(y_hat)
        male = self.classifier_male(y_hat)
        smile = self.classifier_smile(y_hat)
        if not self.dual_direction:
            return (self.loss_f(pose, label[:, 0]) + \
                    self.loss_f(eye_glasses, label[:, 1]) + \
                    self.loss_f(young, label[:, 2]) + \
                    self.loss_f(male, label[:, 3]) + \
                    self.loss_f(smile, label[:, 4]))
        else:
            return (self.loss_f(pose, label[:, 0]) + \
                    self.loss_f(1 - pose, label[:, 1]) + \
                    self.loss_f(eye_glasses, label[:, 2]) + \
                    self.loss_f(1 - eye_glasses, label[:, 3]) + \
                    self.loss_f(young, label[:, 4]) + \
                    self.loss_f(1 - young, label[:, 5]) + \
                    self.loss_f(male, label[:, 6]) + \
                    self.loss_f(1 - male, label[:, 7]) + \
                    self.loss_f(smile, label[:, 8]) + \
                    self.loss_f(1 - smile, label[:, 9]))

        # pose = self.softmax(self.classifier_pose(y_hat))[:, 1]
        # eye_glasses = self.softmax(self.classifier_eyeglasses(y_hat))[:, 1]
        # young = self.softmax(self.classifier_young(y_hat))[:, 1]
        # male = self.softmax(self.classifier_male(y_hat))[:, 1]
        # smile = self.softmax(self.classifier_smile(y_hat))[:, 1]

        # return (self.loss_f(pose, label[:, 0]) + \
        #         self.loss_f(eye_glasses, label[:, 1]) + \
        #         self.loss_f(young, label[:, 2]) + \
        #         self.loss_f(male, label[:, 3]) + \
        #         self.loss_f(smile, label[:, 4]))

    def modify_label(self, label):
        for i in range(label.size(0)):
            if label[i] < 0:
                label[i] = 0
            if label[i] > 1:
                label[i] = 1
        return label


    def comp_img(self, x, y_hat_edit, label):
        y_hat_edit = (y_hat_edit + 1) / 2
        y_hat = (x + 1) / 2

        # print(f"gt attributes: {label}")
        # print(f"predict pose: {torch.argmax(self.classifier_pose(y_hat), 1)}")
        # print(f"predict eyeglasses: {torch.argmax(self.classifier_eyeglasses(y_hat), 1)}")
        # print(f"predict young: {torch.argmax(self.classifier_young(y_hat), 1)}")
        # print(f"predict male: {torch.argmax(self.classifier_male(y_hat), 1)}")
        # print(f"predict smile: {torch.argmax(self.classifier_smile(y_hat), 1)}")
        # import ipdb; ipdb.set_trace()
        pose_label = label[:, 0] + self.softargmax1d(self.classifier_pose(y_hat))
        eye_glasses_label = label[:, 1] + self.softargmax1d(self.classifier_eyeglasses(y_hat))
        young_label = label[:, 2] + self.softargmax1d(self.classifier_young(y_hat))
        male_label = label[:, 3] + self.softargmax1d(self.classifier_male(y_hat))
        smile_label = label[:, 4] + self.softargmax1d(self.classifier_smile(y_hat))

        pose = self.classifier_pose(y_hat_edit)
        eye_glasses = self.classifier_eyeglasses(y_hat_edit)
        young = self.classifier_young(y_hat_edit)
        male = self.classifier_male(y_hat_edit)
        smile = self.classifier_smile(y_hat_edit)

        # if pose_label < 0:
        #     pose_label = 0
        # pose_label =
        # label = label.long()
        # print("pose", pose)
        # print("pose_label", pose_label)
        # print("self.classifier_pose(y_hat)", self.classifier_pose(y_hat))

        return (self.loss_f(pose, self.modify_label(pose_label).long()) + \
                self.loss_f(eye_glasses, self.modify_label(eye_glasses_label).long()) + \
                self.loss_f(young, self.modify_label(young_label).long()) + \
                self.loss_f(male, self.modify_label(male_label).long()) + \
                self.loss_f(smile, self.modify_label(smile_label).long()))

        # pose = self.softmax(self.classifier_pose(y_hat))[:, 1]
        # eye_glasses = self.softmax(self.classifier_eyeglasses(y_hat))[:, 1]
        # young = self.softmax(self.classifier_young(y_hat))[:, 1]
        # male = self.softmax(self.classifier_male(y_hat))[:, 1]
        # smile = self.softmax(self.classifier_smile(y_hat))[:, 1]

        # return (self.loss_f(pose, label[:, 0]) + \
        #         self.loss_f(eye_glasses, label[:, 1]) + \
        #         self.loss_f(young, label[:, 2]) + \
        #         self.loss_f(male, label[:, 3]) + \
        #         self.loss_f(smile, label[:, 4]))




class FeatureExtraction(nn.Module):
    def __init__(self, pretrained, model_type = "Resnet18"):
        super(FeatureExtraction, self).__init__()
        self.model = resnet18(pretrained=pretrained)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
    def forward(self, image):
        return self.model(image)

class FeatureClassfier(nn.Module):
    def __init__(self, selected_attrs,input_dim=512, output_dim = 1):
        super(FeatureClassfier, self).__init__()

        self.attrs_num = len(selected_attrs)
        self.selected_attrs = selected_attrs
        output_dim = len(selected_attrs)
        """build full connect layers for every attribute"""
        self.fc_set = {}

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        res = self.fc(x)
        res = self.sigmoid(res)
        return res

class FaceAttrModel(nn.Module):
    def __init__(self, model_type="Resnet18", pretrained=True, selected_attrs=ATTRIBUTES):
        super(FaceAttrModel, self).__init__()
        self.featureExtractor = FeatureExtraction(pretrained, model_type)
        if model_type == "Resnet18":
            self.featureClassfier = FeatureClassfier(selected_attrs, input_dim=512)
        else:
            self.featureClassfier = FeatureClassfier(selected_attrs, input_dim=2048)

    def forward(self, image):
        features = self.featureExtractor(image)
        results = self.featureClassfier(features)
        return results




class FocalLoss(nn.Module):
    def __init__(self,):
        super(FocalLoss, self).__init__()
        self.focal_loss_alpha = 0.8
        self.focal_loss_gamma = 2
        self.size_average = False

    def forward(self, inputs, targets):
        gpu_targets = targets.cuda()
        alpha_factor = torch.ones(gpu_targets.shape).cuda() * self.focal_loss_alpha
        alpha_factor = torch.where(torch.eq(gpu_targets, 1), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(gpu_targets, 1), 1. - inputs, inputs)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.focal_loss_gamma)
        targets = targets.type(torch.FloatTensor)
        inputs = inputs.cuda()
        targets = targets.cuda()
        bce = F.binary_cross_entropy(inputs, targets)
        focal_weight = focal_weight.cuda()
        cls_loss = focal_weight * bce
        return cls_loss.sum()

class AttrPredLoss_40(nn.Module):
    def __init__(self, pretrained_model_path=None, dual_direction=False):
        super(AttrPredLoss_40, self).__init__()
        model_path = "Resnet18.pth"
        self.classifier = FaceAttrModel().eval()
        self.classifier.load_state_dict(
            torch.load(os.path.join(pretrained_model_path, model_path), map_location=torch.device('cuda')), strict=True)


        self.loss_f = FocalLoss()
        # F.binary_cross_entropy()
        # self.loss_f = nn.CrossEntropyLoss().to('cuda')
        # self.dual_direction = dual_direction
        # self.softmax = nn.Softmax(dim=1).to('cuda')
        # self.loss_f = nn.MSELoss().to('cuda')

        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.selected_attrs_ind = [4, 8, 9, 11, 13, 14, 15, 16, 17, 18, 20, 22, 23, 26, 29, 30, 31, 36]
        print(f'Loading classifier from {model_path}')

    def forward(self, x, label):
        # y_hat = self.face_pool(x)
        # y_hat = (y_hat + 1) / 2
        y_hat = x

        label = label.long()
        pred_all = self.classifier(y_hat)
        # pred = torch.FloatTensor([pred_all[:, j] for j in self.selected_attrs_ind]).cuda()

        loss = 0
        for i in range(x.size(0)):
            pred = torch.FloatTensor([pred_all[i, j] for j in self.selected_attrs_ind]).cuda()
            loss += self.loss_f(pred, label[i])
            # print(f"--> gt attributes: {label[i]>0.5}")
            # print(f"predicts: {pred>0.5}")

        # print(f"predicts: {torch.argmax(self.classifier(y_hat), 1)}")
        # import ipdb; ipdb.set_trace()
        # print(f"--> end")

        # for i in len()
        return loss


    def modify_label(self, label):
        for i in range(label.size(0)):
            if label[i] < 0:
                label[i] = 0
            if label[i] > 1:
                label[i] = 1
        return label


    def softargmax1d(self, input, beta=100):
        *_, n = input.shape
        input = nn.functional.softmax(beta * input, dim=-1)
        indices = torch.linspace(0, 1, n).to('cuda')
        result = torch.sum((n - 1) * input * indices, dim=-1)
        return result


    def comp_img(self, x, y_hat_edit, label):
        ## not finished, bugs in here
        y_hat_edit = (y_hat_edit + 1) / 2
        y_hat = (x + 1) / 2

        # y_hat = x
        print("---------->")
        print("0: ", label.size())
        print("01: ", self.classifier(y_hat).size(), self.classifier(y_hat))
        print("02: ", self.softargmax1d(self.classifier(y_hat)).size())

        label = label + self.softargmax1d(self.classifier(y_hat))
        print("---------->")
        print("1: ", label)
        label = self.modify_label(label).long()
        print("2: ", label)
        pred_all = self.classifier(y_hat_edit)

        loss = 0
        for i in range(x.size(0)):
            pred = torch.FloatTensor([pred_all[i, j] for j in self.selected_attrs_ind]).cuda()
            print("3: ", pred.size(), pred)
            print("4: ", label[i].size(), label[i])

            loss += self.loss_f(pred, label[i])
        return loss

        #
        # # import ipdb; ipdb.set_trace()
        # pose_label = label[:, 0] + self.softargmax1d(self.classifier_pose(y_hat))
        # eye_glasses_label = label[:, 1] + self.softargmax1d(self.classifier_eyeglasses(y_hat))
        # young_label = label[:, 2] + self.softargmax1d(self.classifier_young(y_hat))
        # male_label = label[:, 3] + self.softargmax1d(self.classifier_male(y_hat))
        # smile_label = label[:, 4] + self.softargmax1d(self.classifier_smile(y_hat))
        #
        # pose = self.classifier_pose(y_hat_edit)
        # eye_glasses = self.classifier_eyeglasses(y_hat_edit)
        # young = self.classifier_young(y_hat_edit)
        # male = self.classifier_male(y_hat_edit)
        # smile = self.classifier_smile(y_hat_edit)
        #
        #
        # return (self.loss_f(pose, self.modify_label(pose_label).long()) + \
        #         self.loss_f(eye_glasses, self.modify_label(eye_glasses_label).long()) + \
        #         self.loss_f(young, self.modify_label(young_label).long()) + \
        #         self.loss_f(male, self.modify_label(male_label).long()) + \
        #         self.loss_f(smile, self.modify_label(smile_label).long()))
        #
