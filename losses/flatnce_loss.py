import torch
import torch.nn as nn
# Last Change:  2022-09-09 12:14:41
import numpy as np
import torch.nn.functional as F

class FlatNceLoss(nn.Module):

    def __init__(self, batch_size, emb_dim=512):
        super(FlatNceLoss, self).__init__()

        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.batch_size = batch_size
        self.logit_scale = 20.0
        self.n_views = 2
        self.loss = nn.CrossEntropyLoss()
        self.K = int(batch_size * 10)
        self.query_feats_bank = torch.zeros(self.K, emb_dim).cuda()
        self.doc_feats_bank = torch.zeros(self.K, emb_dim).cuda()
        self.ptr = 0
        self.is_full = False

    def enqueue_dequeue(self, query_feat, doc_feat):
        q_size = query_feat.shape[0]
        if self.ptr + q_size > self.K:
            self.ptr = q_size
            self.is_full = True

        tmp_query = self.query_feats_bank[0 : q_size]
        tmp_doc = self.doc_feats_bank[0 : q_size]
        self.query_feats_bank[self.ptr : self.ptr + q_size] = tmp_query
        self.doc_feats_bank[self.ptr : self.ptr + q_size] = tmp_doc
        self.query_feats_bank[0 : q_size] = query_feat
        self.doc_feats_bank[0 : q_size] = doc_feat
        self.ptr += q_size

    def get(self):
        if self.is_full:
            return self.query_feats_bank, self.doc_feats_bank
        else:
            return self.query_feats_bank[:self.ptr], self.doc_feats_bank[:self.ptr]

    def _flatnce(self, features):
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0).cuda()
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)

        labels = torch.zeros(positives.shape[0], dtype=torch.long).cuda()

        #        logits = logits / self.args.temperature #- 
        logits = self.logit_scale * (negatives - positives) # (512,510) #+
        
        v = torch.logsumexp(logits, dim=1, keepdim=True)
        loss_vec = torch.exp(v-v.detach())
        tmp_logit = torch.zeros(logits.size(0),1).cuda()
        dummy_logits = torch.cat([tmp_logit, logits],1)
        loss = loss_vec.mean()-1 + self.loss(dummy_logits, labels).detach()
        return loss

    def forward(self, image_emb, text_emb, tb_tools, is_xbm=False):
        
        features1 = torch.cat([image_emb, text_emb], dim=0)
        features2 = torch.cat([text_emb, image_emb], dim=0)
        
        loss1 = self._flatnce(features1)
        loss2 = self._flatnce(features2)

        return (loss1 + loss2) / 2


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
