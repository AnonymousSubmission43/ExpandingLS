import torch
from torch import nn
import torch.nn.functional as F
from configs.paths_config import model_paths


class DisentangleLoss(nn.Module):

    def __init__(self, device='cuda'):
        super(DisentangleLoss, self).__init__()
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def info_nce_loss(self, feat_enc, feat_gt):
        features = torch.cat([feat_enc, feat_gt], 0)
        features = F.normalize(features, dim=1)

        labels = torch.cat([torch.arange(feat_enc.size(0)) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # logits = logits / self.args.temperature

        return logits, labels

    def forward(self, feat_enc, feat_gt):
        logits, labels = self.info_nce_loss(feat_enc, feat_gt)
        loss = self.criterion(logits, labels)
        return loss
