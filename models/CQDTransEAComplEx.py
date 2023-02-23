from typing import Tuple
from models.CQDBaseModel import CQDBaseModel

import torch
import torch.nn as nn
from torch import Tensor


class CQDTransEAComplEx(CQDBaseModel):
    """
    ComplEx + Attribute Model
    """

    def __init__(self,
                 use_attributes=True,
                 do_sigmoid=True,
                 use_modulus=False,
                 *args,
                 **kwargs
                 ):
        super(CQDTransEAComplEx, self).__init__(*args, **kwargs)

        self.use_attributes = use_attributes
        self.do_sigmoid = do_sigmoid

        self.ent_embeddings = nn.Embedding(self.nentity, 2*self.rank)
        self.rel_embeddings = nn.Embedding(self.nrelation, 2*self.rank)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        if self.use_attributes:
            self.attr_embeddings = nn.Embedding(self.nattr, 2*self.rank)
            nn.init.xavier_uniform_(self.attr_embeddings.weight.data)
            self.b = nn.Embedding(self.nattr, 1 * 2)

        self.use_modulus = use_modulus

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

    def score_rel(self, data):
        h = self.ent_embeddings(data['batch_h'])
        r = self.rel_embeddings(data["batch_r"])
        t = self.ent_embeddings(data['batch_t'])
        (h_re, h_im), (r_re, r_im), (t_re, t_im) = self.split(h, r, t)
        batch_size = h.shape[0]
        score_1 = (h_re * r_re - h_im * r_im).view(batch_size, 1, self.rank) @ t_re.view(batch_size, self.rank, 1)
        score_2 = (h_im * r_re + h_re * r_im).view(batch_size, 1, self.rank) @ t_im.view(batch_size, self.rank, 1)
        scores = (score_1 + score_2).view(batch_size)
        return scores

    def score_attr(self, data):
        if "batch_e" not in data:
            return torch.FloatTensor().to(device=data['batch_h'].device)
        e = self.ent_embeddings(data['batch_e'])
        v = data['batch_v']
        preds = self.predict_attribute_values(e, data['batch_a'])
        return torch.stack((preds, v), dim=-1)

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

        if self.use_modulus:
            predictions = torch.sqrt(torch.square(re_pred) + torch.square(im_pred))
        else:
            predictions = (re_pred + im_pred) / 2
        return predictions

    def score_o_all(self, lhs_emb: Tensor, rel_emb: Tensor, rhs_emb: Tensor):
        return self.score_o(lhs_emb, rel_emb, rhs_emb)

    def score_o(self, lhs_emb: Tensor, rel_emb: Tensor, rhs_emb: Tensor):
        (h_re, h_im), (r_re, r_im), (t_re, t_im) = self.split(lhs_emb, rel_emb, rhs_emb)

        score_1 = (h_re * r_re - h_im * r_im) @ t_re.transpose(-1, -2)
        score_2 = (h_im * r_re + h_re * r_im) @ t_im.transpose(-1, -2)
        scores = score_1 + score_2
        return scores

    #====================== Methods required for Link Prediction Evaluation ======================#
    def score_s_all(self,
                    lhs_emb: Tensor,
                    rel_emb: Tensor,
                    rhs_emb: Tensor):
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        score_1 = (rhs[0] * rel[0] + rhs[1] * rel[1]) @ lhs[0].transpose(-1, -2)
        score_2 = (rhs[1] * rel[0] - rhs[0] * rel[1]) @ lhs[1].transpose(-1, -2)
        return score_1 + score_2

    def score_candidates(self, triples: Tensor):
        lhs_emb = self.ent_embeddings(triples[:, 0])
        rel_emb = self.rel_embeddings(triples[:, 1])
        rhs_emb = self.ent_embeddings(triples[:, 2])
        to_score = self.ent_embeddings.weight
        scores_o = self.score_o_all(lhs_emb, rel_emb, to_score)
        scores_s = self.score_s_all(to_score, rel_emb, rhs_emb)
        return (scores_o, scores_s), None
