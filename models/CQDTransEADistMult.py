from models.CQDBaseModel import CQDBaseModel

import torch
import torch.nn as nn
from torch import Tensor


class CQDTransEADistMult(CQDBaseModel):
    """
    DistMult + Attribute Model
    """

    def __init__(self,
                 use_attributes=True,
                 do_sigmoid=True,
                 *args,
                 **kwargs
                 ):
        super(CQDTransEADistMult, self).__init__(*args, **kwargs)

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

    def score_rel(self, data):
        h = self.ent_embeddings(data['batch_h'])
        r = self.rel_embeddings(data["batch_r"])
        t = self.ent_embeddings(data['batch_t'])

        scores = (h * r * t).sum(dim=-1)
        return scores

    def score_attr(self, data):
        if "batch_e" not in data:
            return torch.FloatTensor().to(device=data['batch_h'].device)
        e = self.ent_embeddings(data['batch_e'])
        v = data['batch_v']
        preds = self.predict_attribute_values(e, data['batch_a'])
        return torch.stack((preds, v), dim=-1)

    def score(self, data):
        rel_score = self.score_rel(data)
        return rel_score, self.score_attr(data)

    def predict_attribute_values(self, e_emb, attributes):
        # Predict attribute values for a batch
        # e_emb: [B, E], attributes: [B]
        # returns [B]
        a = self.attr_embeddings(attributes)
        b = self.b(attributes).view(attributes.shape[0])
        predictions = torch.sum(e_emb * a, dim=-1) + b
        if self.do_sigmoid:
            predictions = torch.sigmoid(predictions)
        return predictions

    def score_o_all(self,
                    lhs_emb: Tensor,
                    rel_emb: Tensor,
                    rhs_emb: Tensor):
        return self.score_o(lhs_emb, rel_emb, rhs_emb)

    def score_o(self,
                lhs_emb: Tensor,
                rel_emb: Tensor,
                rhs_emb: Tensor):
        scores = (lhs_emb * rel_emb) @ rhs_emb.transpose(-1, -2)
        return scores
