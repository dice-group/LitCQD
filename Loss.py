from typing import Tuple
import torch
import torch.nn.functional as F
from torch.functional import Tensor


class Loss(object):
    def __init__(self, neg_size):
        self.neg_size = neg_size

    def split(self, batch_score: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = self._compute_batch_size(batch_score)
        p_score = batch_score[:batch_size]
        n_score = batch_score[batch_size:]
        return p_score, n_score

    def compute(self, batch_score: Tensor, subsampling_weight: Tensor) -> Tensor:
        raise NotImplementedError()

    def _compute_batch_size(self, batch_score: Tensor):
        # If dataloader returns a batch smaller than the usual batch_size
        # Usually at the end of an epoch
        # Alternative: set drop_last=True for PyTorch Dataloader
        return batch_score.shape[0]//(1+self.neg_size)


class MRLoss(Loss):
    # margin ranking loss (TransE)
    def __init__(self, margin=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin

    def compute(self, batch_score: Tensor, subsampling_weight: Tensor) -> Tensor:
        batch_score = - batch_score
        batch_size = self._compute_batch_size(batch_score)
        p_score = batch_score[:batch_size].view(-1, batch_size).permute(1, 0)
        n_score = batch_score[batch_size:].view(-1, batch_size).permute(1, 0)
        return torch.sum(F.relu(p_score - n_score + self.margin)) / batch_score.shape[0]


class Q2BLoss(Loss):
    # negative sampling loss implemented by Query2Box
    def __init__(self, margin=24.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin

    def compute(self, batch_score: Tensor, subsampling_weight: Tensor) -> Tensor:
        p_score, n_score = self.split(batch_score)
        p_score = F.logsigmoid(self.margin - p_score)
        n_score = F.logsigmoid(-(self.margin - n_score))

        # required to get correct margin; get mean neg_score per pos sample:
        n_scores = torch.split(n_score, self.neg_size)
        n_scores = [x.mean() for x in n_scores]
        n_score = torch.stack(n_scores)

        # apply subsampling weight
        p_score = (subsampling_weight * p_score).sum() / subsampling_weight.sum()
        n_score = (subsampling_weight * n_score).sum() / subsampling_weight.sum()

        return - (p_score + n_score) / 2


class CELoss(Loss):
    # See https://github.com/uma-pi1/kge/blob/db908a99df5efe20f960dc3cf57eb57206c2f36c/kge/util/loss.py#L192
    # for KLDivLoss function use
    # and https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580
    # for one-hot targets thread
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.KLDivLoss(reduction='mean')

    def compute(self, batch_score: Tensor, subsampling_weight: Tensor) -> Tensor:
        pos_samples_size = self._compute_batch_size(batch_score)

        def per_sample_loss():
            nonlocal batch_score
            # reorder batch_score
            # from (pos_1, ..., pos_n, neg_1_1, ..., neg_1_x, ..., neg_n_x)
            # to (pos_1, neg_1_1, ..., neg_1_x, pos_2, ..., neg_n_x)
            indices = list()
            for i in range(pos_samples_size):
                indices.extend([i, *(pos_samples_size+i+j for j in range(self.neg_size))])
            batch_score = batch_score[torch.LongTensor(indices)]
            # compute log-softmax per sample
            batch_score = batch_score.reshape((pos_samples_size, -1))
            temp = F.log_softmax(batch_score, dim=-1)
            temp = temp[:, 0]  # only the value for the positive sample is relevant
            return - sum(temp) / len(temp)
            # returns the same as:
            #labels = torch.tensor([[0]*batch_score.shape[-1]]*batch_score.shape[0])
            #labels[..., 0] = 1
            # return self.loss_fn(F.log_softmax(batch_score, dim=-1), F.normalize(labels.float(), p=1, dim=1))

        def per_batch_loss():
            temp = F.log_softmax(batch_score, dim=-1)
            return - torch.sum(temp[:pos_samples_size], dim=-1) / pos_samples_size

            # alternatively use kldivLoss:
            target = torch.cat((torch.ones(pos_samples_size), torch.zeros(pos_samples_size*self.neg_size))).cuda()
            # target/||target||_1 -> target = target/pos_sample_size
            target = F.normalize(target, p=1, dim=-1)
            return self.loss_fn(temp, target)

        return per_sample_loss()
        # return per_batch_loss()


class LimitLoss(Loss):
    # Set an upper limit for positive score
    # http://zhuqiannan.com/files/CIKM2017_Scoring_Loss.pdf
    def __init__(self, limit=6.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit

    def compute(self, batch_score: Tensor, subsampling_weight: Tensor) -> Tensor:
        return torch.sum(F.relu(batch_score - self.limit))


class RSLoss(Loss):
    # Combination of limit- and margin-based loss
    # http://zhuqiannan.com/files/CIKM2017_Scoring_Loss.pdf
    def __init__(self, margin=2.0, limit=4.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.limit = limit

    def compute(self, batch_score: Tensor, subsampling_weight: Tensor) -> Tensor:
        p_score, n_score = self.split(batch_score)
        return torch.sum(F.relu(p_score - n_score + self.margin) + F.relu(p_score - self.limit))


class MAELoss(Loss):
    # mean absolute error for attributes
    def compute(self, batch_score: Tensor, subsampling_weight: Tensor = None) -> Tensor:
      
      return torch.mean(abs(batch_score[..., 0] - batch_score[..., 1]))


class MSELoss(Loss):
    # mean squared error for attributes
    def compute(self, batch_score: Tensor, subsampling_weight: Tensor = None) -> Tensor:
        return torch.mean((batch_score[..., 0] - batch_score[..., 1])**2)
