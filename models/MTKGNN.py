from torch.nn.init import xavier_normal_

import torch
import torch.nn as nn
from torch import Tensor

from typing import Dict, Tuple


class MTKGNN(nn.Module):
    """
    TransE + Attribute Model of MTKGNN
    """

    def __init__(self,
                 nentity,
                 nrelation,
                 nattr,
                 rank,
                 p_norm,
                 ):
        super(MTKGNN, self).__init__()

        self.nentity = nentity
        self.nrelation = nrelation
        self.nattr = nattr
        self.rank = rank

        self.ent_embeddings = torch.nn.Embedding(self.nentity, self.rank)
        self.rel_embeddings = torch.nn.Embedding(self.nrelation, self.rank)

        self.attr_embeddings = torch.nn.Embedding(self.nattr, self.rank)

        self.attr_net_left = torch.nn.Sequential(
            torch.nn.Linear(2*self.rank, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 1))

        self.p_norm = p_norm

        xavier_normal_(self.ent_embeddings.weight.data)
        xavier_normal_(self.rel_embeddings.weight.data)

    def _calc(self, h, r, t):
        score = (h + r) - t
        return torch.norm(score, self.p_norm, -1).flatten()

    def score_rel(self, data):
        h = self.ent_embeddings(data['batch_h'])
        r = self.rel_embeddings(data['batch_r'])
        t = self.ent_embeddings(data['batch_t'])
        return self._calc(h, r, t)

    #---------- Overridden methods ----------#

    def predict_attribute_values(self, e_emb, attributes):
        a = self.attr_embeddings(attributes)
        inputs = torch.cat([e_emb, a], dim=1)
        pred = self.attr_net_left(inputs)
        return pred.view(attributes.shape)

    def score(self, data):
        score_rel = -self.score_rel(data)
        score_attr = torch.stack(((self.predict_attribute_values(self.ent_embeddings(data['batch_e']), data['batch_a'])), data['batch_v']), dim=-1)
        return score_rel, score_attr

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

    def forward(self, batch_queries_dict: Dict[Tuple, Tensor]):
        all_scores = []

        for query_structure, queries in batch_queries_dict.items():
            if query_structure == ('e', ('r',)):
                pred = self.score_o_all(self.ent_embeddings(queries[:, 0].long()).unsqueeze(1), self.rel_embeddings(queries[:, 1].long()).unsqueeze(1), self.ent_embeddings.weight).squeeze(1)
                all_scores.extend([s for s in pred])
            elif query_structure == ('e', ('ap', 'a')):
                pred = self.predict_attribute_values(self.ent_embeddings(queries[:, 0].long()), queries[:, 2].long())
                all_scores.extend([s for s in pred])
        return all_scores
