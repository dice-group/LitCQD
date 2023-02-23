from models.CQDBaseModel import CQDBaseModel

import torch
import torch.nn as nn
from torch import Tensor

from typing import List, Optional, Tuple


class N3:
    def __init__(self, weight: float):
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class CQDTransComplExA(CQDBaseModel):
    """
    TransComplEx + Attribute Model
    """

    def __init__(self,
                 do_sigmoid=True,
                 p_norm=2,
                 use_attributes=True,
                 reg_weight=1e-2,
                 *args,
                 **kwargs
                 ):
        super(CQDTransComplExA, self).__init__(*args, **kwargs)

        self.do_sigmoid = do_sigmoid
        self.p_norm = p_norm
        self.use_attributes = use_attributes

        self.ent_embeddings = nn.Embedding(self.nentity, self.rank*2)
        self.rel_embeddings = nn.Embedding(self.nrelation, self.rank*2)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.regularizer = N3(reg_weight)

        if self.use_attributes:
            self.attr_embeddings = nn.Embedding(self.nattr, self.rank*2)
            nn.init.xavier_uniform_(self.attr_embeddings.weight.data)
            self.b = nn.Embedding(self.nattr, 1 * 2)

    def split(self,
              lhs_emb: Tensor,
              rel_emb: Tensor,
              rhs_emb: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Split embeddings into their real and imaginary parts.
        """
        lhs = lhs_emb[..., :self.rank], lhs_emb[..., self.rank:]
        rel = rel_emb[..., :self.rank], rel_emb[..., self.rank:]
        rhs = rhs_emb[..., :self.rank], rhs_emb[..., self.rank:]
        return lhs, rel, rhs

    def get_factors(self,
                    lhs: Tuple[Tensor, Tensor],
                    rel: Tuple[Tensor, Tensor],
                    rhs: Tuple[Tensor, Tensor]) -> List[Tensor]:
        factors = []
        for term in (lhs, rel, rhs):
            factors.append(torch.sqrt(term[0] ** 2 + term[1] ** 2))
        return factors

    def _calc(self, h, r, t, mode):
        (re_head, im_head), (re_rel, im_rel), (re_tail, im_tail) = self.split(h, r, t)
        re_score = (re_head + re_rel) - re_tail
        im_score = (im_head + im_rel) + im_tail

        normscore_re = torch.norm(re_score, p=self.p_norm, dim=-1)
        normscore_im = torch.norm(im_score, p=self.p_norm, dim=-1)
        return (normscore_re + normscore_im)

    def score_rel(self, data):
        h = self.ent_embeddings(data['batch_h'])
        r = self.rel_embeddings(data['batch_r'])
        t = self.ent_embeddings(data['batch_t'])
        return self._calc(h, r, t, data["mode"])

    def score_attr(self, data):
        if "batch_e" not in data:
            return torch.FloatTensor().cuda()
        e = self.ent_embeddings(data['batch_e'])
        v = data['batch_v']
        attr_pred = self.predict_attribute_values(e, data['batch_a'])
        return torch.stack((attr_pred, v), dim=-1)

    #---------- Overridden methods ----------#

    def score(self, data):
        return self.score_rel(data), self.score_attr(data)

    def predict_attribute_values(self, e_emb, attributes):
        # Predict attribute values for a batch
        # e_emb: [B, E], attributes: [B]
        # returns [B]
        re_head, im_head = torch.chunk(e_emb, 2, dim=-1)
        re_attr, im_attr = torch.chunk(self.attr_embeddings(attributes), 2, dim=-1)
        re_b, im_b = torch.chunk(self.b(attributes), 2, dim=-1)
        re = re_attr * re_head - im_attr * im_head
        im = re_attr * im_head + im_attr * re_head

        re_pred = torch.sum(re, dim=-1) + re_b.view(attributes.shape[0])
        im_pred = torch.sum(im, dim=-1) + im_b.view(attributes.shape[0])
        if self.do_sigmoid:
            re_pred = torch.sigmoid(re_pred)
            im_pred = torch.sigmoid(im_pred)

        predictions = (re_pred + im_pred) / 2
        return predictions

    def score_o(self,
                lhs_emb: Tensor,
                rel_emb: Tensor,
                rhs_emb: Tensor):
        """
        [batch_size, emb_size] or
        [batch_size, var, emb_size]
        Returns [batch_size] or
        """
        (re_head, im_head), (re_rel, im_rel), (re_tail, im_tail) = self.split(lhs_emb, rel_emb, rhs_emb)

        re_predicted = (re_head + re_rel)
        im_predicted = (im_head + im_rel)

        re_score = torch.norm(re_predicted - re_tail, self.p_norm, -1)
        im_score = torch.norm(im_predicted + im_tail, self.p_norm, -1)

        return (re_score + im_score)

    def score_o_all(self,
                    lhs_emb: Tensor,
                    rel_emb: Tensor,
                    rhs_emb: Tensor):
        """
        [batch_size, var, emb_size] for lhs,rel and
        [nentity, emb_size] for rhs
        Returns [batch_size, var, nentity]
        """
        rhs_emb = rhs_emb.unsqueeze(0)

        (re_head, im_head), (re_rel, im_rel), (re_tail, im_tail) = self.split(lhs_emb, rel_emb, rhs_emb)

        scores = []
        with torch.no_grad():
            for i in range(rel_emb.shape[0]):
                score_1 = torch.norm(re_head[i] + re_rel[i] - re_tail, p=self.p_norm, dim=-1)
                score_2 = torch.norm(im_head[i] + im_rel[i] + im_tail, p=self.p_norm, dim=-1)
                scores.append(score_1+score_2)

        return torch.stack(scores)
