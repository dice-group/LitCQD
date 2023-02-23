from models.CQDBaseModel import CQDBaseModel

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class CQDTransEA(CQDBaseModel):
    """
    TransE + Attribute Model
    """

    def __init__(self,
                 p_norm=1,
                 use_attributes=True,
                 do_sigmoid=False,
                 *args,
                 **kwargs
                 ):
        super(CQDTransEA, self).__init__(*args, **kwargs)

        self.p_norm = p_norm
        self.use_attributes = use_attributes
        self.do_sigmoid = do_sigmoid

        self.ent_embeddings = nn.Embedding(self.nentity, self.rank)
        self.rel_embeddings = nn.Embedding(self.nrelation, self.rank)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        if self.use_attributes:
            self.attr_embeddings = nn.Embedding(self.nattr, self.rank)
            nn.init.xavier_uniform_(self.attr_embeddings.weight.data)
            self.b = nn.Embedding(self.nattr, 1)

    def _calc(self, h, r, t):
        score = (h + r) - t
        return torch.norm(score, self.p_norm, -1).flatten()

    def score_rel(self, data):
        h = self.ent_embeddings(data['batch_h'])
        r = self.rel_embeddings(data['batch_r'])
        t = self.ent_embeddings(data['batch_t'])
        return self._calc(h, r, t)

    def score_attr(self, data):
        if "batch_e" not in data:
            return torch.FloatTensor().to(device=data['batch_h'].device)
        e = self.ent_embeddings(data['batch_e'])
        v = data['batch_v']
        preds = self.predict_attribute_values(e, data['batch_a'])
        return torch.stack((preds, v), dim=-1)

    def score(self, data):
        return -self.score_rel(data), self.score_attr(data)

    def predict_attribute_values(self, e_emb, attributes):
        # Predict attribute values for a batch
        # e_emb: [B, E], attributes: [B]
        # returns [B]
        a = self.attr_embeddings(attributes)
        b = self.b(attributes).squeeze()
        predictions = torch.sum(e_emb * a, dim=-1) + b
        if self.do_sigmoid:
            predictions = torch.sigmoid(predictions)
        return predictions

    def score_o(self,
                lhs_emb: Tensor,
                rel_emb: Tensor,
                rhs_emb: Tensor):

        predicted = (lhs_emb + rel_emb)

        return - torch.norm(predicted - rhs_emb, self.p_norm, -1)

    def score_o_all(self,
                    lhs_emb: Tensor,
                    rel_emb: Tensor,
                    rhs_emb: Tensor):

        predicted = (lhs_emb + rel_emb)

        scores = torch.empty(size=(lhs_emb.shape[0], lhs_emb.shape[1], rhs_emb.shape[0])).to(device=lhs_emb.device)
        for ent in range(lhs_emb.shape[0]):
            for var in range(lhs_emb.shape[1]):
                scores[ent, var] = - torch.norm(predicted[ent, var] - rhs_emb, self.p_norm, -1)

        return scores
