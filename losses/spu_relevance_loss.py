import torch
import torch.nn as nn
# Last Change:  2022-09-22 10:36:28
import numpy as np


class SpuRelevanceLoss(nn.Module):

    def __init__(self, batch_size, emb_dim=512):
        super(SpuRelevanceLoss, self).__init__()

        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = 20.0
        self.loss_img = nn.BCEWithLogitsLoss()
        self.loss_txt = nn.BCEWithLogitsLoss()
        self.loss_1 = nn.BCEWithLogitsLoss()
        self.loss_2 = nn.BCEWithLogitsLoss()
        #self.ground_truth = torch.arange(batch_size).cuda()

        self.K = int(batch_size * 10)
        #self.query_feats_bank = torch.zeros(self.K, emb_dim).cuda()
        #self.doc_feats_bank = torch.zeros(self.K, emb_dim).cuda()
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

    def forward(self, image_emb, text_emb, spu_ids, local_rank, tb_tools, is_xbm=False):
        #assert torch.isnan(image_emb).sum() == 0, "img emb " + str(image_emb)
        #assert torch.isnan(text_emb).sum() == 0, "txt emb" + str(text_emb)
        logit_scale = self.logit_scale
        logits_per_image = logit_scale * image_emb @ text_emb.t()
        logits_per_text = logit_scale * text_emb @ image_emb.t()
        #assert torch.isnan(logits_per_image).sum() == 0, "img error " + str(logits_per_image)
        #assert torch.isnan(logits_per_text).sum() == 0, "txt error " + str(logits_per_text)
        
        label = torch.from_numpy(np.array(spu_ids).astype(np.float64)).cuda(local_rank)
        batch = label.shape[0]
        label = label.expand((batch, batch))
        ground_truth = (label == label.t()).float()
        loss_img = self.loss_img(logits_per_image, ground_truth)
        loss_txt = self.loss_txt(logits_per_text, ground_truth)

        #assert torch.isnan(loss_img).sum() == 0, "img final loss error " + str(loss_img) + str(logits_per_image.max()) + str(logits_per_image.min()) + str(logits_per_text.max()) + str(logits_per_text.min())
        #assert torch.isnan(loss_txt).sum() == 0, "txt final loss error " + str(loss_txt) + str(logits_per_image.max()) + str(logits_per_image.min()) + str(logits_per_text.max()) + str(logits_per_text.min())
        '''
        if is_xbm:
            self.enqueue_dequeue(image_emb.detach(), text_emb.detach())
            query_bank, doc_bank = self.get()

            logits_1 = logit_scale * image_emb @ doc_bank.t() # N * K
            loss_1 = self.loss_1(logits_1, ground_truth)

            logits_2 = logit_scale * text_emb @ query_bank.t()
            loss_2 = self.loss_2(logits_2, ground_truth)
        
        if is_xbm:
            total_loss = (loss_img + loss_txt + loss_1 + loss_2) / 4
        else:
            total_loss = (loss_img + loss_txt) / 2
        '''
        total_loss = (loss_img + loss_txt) / 2

        '''
        if tb_tools['local_rank'] == 0:
            acc1_img, acc5_img = accuracy(logits_per_image, self.ground_truth, topk=(1, 5))
            acc1_txt, acc5_txt = accuracy(logits_per_text, self.ground_truth, topk=(1, 5))
            tb_tools['tb_writer'].add_scalar('{}/acc1_img'.format(tb_tools['prefix']), acc1_img[0], global_step=tb_tools['global_step'])
            tb_tools['tb_writer'].add_scalar('{}/acc5_img'.format(tb_tools['prefix']), acc5_img[0], global_step=tb_tools['global_step'])
            tb_tools['tb_writer'].add_scalar('{}/acc1_txt'.format(tb_tools['prefix']), acc1_txt[0], global_step=tb_tools['global_step'])
            tb_tools['tb_writer'].add_scalar('{}/acc5_txt'.format(tb_tools['prefix']), acc5_txt[0], global_step=tb_tools['global_step'])
            
            if is_xbm:
                acc1_img, acc5_img = accuracy(logits_1, self.ground_truth, topk=(1, 5))
                acc1_txt, acc5_txt = accuracy(logits_2, self.ground_truth, topk=(1, 5))
                tb_tools['tb_writer'].add_scalar('{}/acc1_img_with_bank'.format(tb_tools['prefix']), acc1_img[0], global_step=tb_tools['global_step'])
                tb_tools['tb_writer'].add_scalar('{}/acc5_img_with_bank'.format(tb_tools['prefix']), acc5_img[0], global_step=tb_tools['global_step'])
                tb_tools['tb_writer'].add_scalar('{}/acc1_txt_with_bank'.format(tb_tools['prefix']), acc1_txt[0], global_step=tb_tools['global_step'])
                tb_tools['tb_writer'].add_scalar('{}/acc5_txt_with_bank'.format(tb_tools['prefix']), acc5_txt[0], global_step=tb_tools['global_step'])
            '''
        return total_loss


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
