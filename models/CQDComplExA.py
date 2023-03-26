from models.CQDBaseModel import CQDBaseModel

import torch
import torch.nn as nn
from torch import optim, Tensor
import math

from typing import Tuple, List, Optional


class N3:
    def __init__(self, weight: float):
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class CQDComplExA(CQDBaseModel):
    """
    ComplEx-N3 + Attribute Model
    """

    def __init__(self,
                 use_attributes: bool = True,
                 init_size: float = 1e-3,
                 reg_weight: float = 1e-2,
                 *args,
                 **kwargs,
                 ):
        super(CQDComplExA, self).__init__(*args, **kwargs)

        self.use_attributes = use_attributes

        self.ent_embeddings = nn.Embedding(self.nentity, 2*self.rank)
        self.rel_embeddings = nn.Embedding(self.nrelation, 2*self.rank)

        self.ent_embeddings.weight.data *= init_size
        self.rel_embeddings.weight.data *= init_size

        if self.use_attributes:
            self.attr_embeddings = nn.Embedding(self.nattr, 2*self.rank)
            self.b = nn.Embedding(self.nattr, 1 * 2)
            self.attr_embeddings.weight.data *= init_size
            self.b.weight.data *= init_size

        self.init_size = init_size
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.regularizer = N3(reg_weight)

    def split(self,
              lhs_emb: Tensor,
              rel_emb: Tensor,
              rhs_emb: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        lhs = lhs_emb[..., :self.rank], lhs_emb[..., self.rank:]
        rel = rel_emb[..., :self.rank], rel_emb[..., self.rank:]
        rhs = rhs_emb[..., :self.rank], rhs_emb[..., self.rank:]
        return lhs, rel, rhs

    def score_attr(self, data):
        if "batch_e" not in data:
            return torch.FloatTensor().to(device=data['batch_h'].device), []
        e = self.ent_embeddings(data['batch_e'])
        v = data['batch_v']
        preds = self.predict_attribute_values(e, data['batch_a'])

        factors = []
        for emb in (e, self.attr_embeddings(data['batch_a']),):
            emb_re, emb_im = emb[..., :self.rank], emb[..., self.rank:]
            factors.append(torch.sqrt(emb_re ** 2 + emb_im ** 2))

        return torch.stack((preds, v), dim=-1), factors

    def loss(self, data, attr_loss_fn, alpha) -> Tensor:
        triples = torch.cat((data['batch_h'].unsqueeze(1), data['batch_r'].unsqueeze(1), data['batch_t'].unsqueeze(1)), -1)
        (scores_o, scores_s), factors = self.score_candidates(triples)
        l_fit = self.loss_fn(scores_o, triples[:, 2]) + self.loss_fn(scores_s, triples[:, 0])
        l_reg = self.regularizer.forward(factors)

        if not self.use_attributes:
            return (l_fit + l_reg)

        attr_scores, attr_factors = self.score_attr(data)
        attr_loss = attr_loss_fn(attr_scores)
        attr_reg = self.regularizer.forward(attr_factors)

        return (l_fit+l_reg) + (1-alpha) * (attr_loss + attr_reg)

    def score_candidates(self, triples: Tensor) -> Tuple[Tuple[Tensor, Tensor], Optional[List[Tensor]]]:
        lhs_emb = self.ent_embeddings(triples[:, 0])
        rel_emb = self.rel_embeddings(triples[:, 1])
        rhs_emb = self.ent_embeddings(triples[:, 2])
        to_score = self.ent_embeddings.weight
        scores_o = self.score_o_all(lhs_emb, rel_emb, to_score)
        scores_s = self.score_s_all(to_score, rel_emb, rhs_emb)
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        factors = self.get_factors(lhs, rel, rhs)
        return (scores_o, scores_s), factors

    def score_o(self, lhs_emb: Tensor, rel_emb: Tensor, rhs_emb: Tensor):
        (h_re, h_im), (r_re, r_im), (t_re, t_im) = self.split(lhs_emb, rel_emb, rhs_emb)
        batch_size = lhs_emb.shape[0]
        score_1 = (h_re * r_re - h_im * r_im).view(batch_size, 1, self.rank) @ t_re.view(batch_size, self.rank, 1)
        score_2 = (h_im * r_re + h_re * r_im).view(batch_size, 1, self.rank) @ t_im.view(batch_size, self.rank, 1)
        scores = (score_1 + score_2).view(batch_size)
        return scores

    def score_o_all(self,
                    lhs_emb: Tensor,
                    rel_emb: Tensor,
                    rhs_emb: Tensor) -> Tuple[Tensor, Optional[List[Tensor]]]:
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        score_1 = (lhs[0] * rel[0] - lhs[1] * rel[1]) @ rhs[0].transpose(-1, -2)
        score_2 = (lhs[1] * rel[0] + lhs[0] * rel[1]) @ rhs[1].transpose(-1, -2)
        return score_1 + score_2

    def score_s_all(self,
                    lhs_emb: Tensor,
                    rel_emb: Tensor,
                    rhs_emb: Tensor) -> Tuple[Tensor, Optional[List[Tensor]]]:
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        score_1 = (rhs[0] * rel[0] + rhs[1] * rel[1]) @ lhs[0].transpose(-1, -2)
        score_2 = (rhs[1] * rel[0] - rhs[0] * rel[1]) @ lhs[1].transpose(-1, -2)
        return score_1 + score_2

    def get_factors(self,
                    lhs: Tuple[Tensor, Tensor],
                    rel: Tuple[Tensor, Tensor],
                    rhs: Tuple[Tensor, Tensor]) -> List[Tensor]:
        factors = []
        for term in (lhs, rel, rhs):
            factors.append(torch.sqrt(term[0] ** 2 + term[1] ** 2))
        return factors

    def predict_attribute_values_all_unused(self, attributes):
        # Predict attribute values for all entities for a batch of attributes
        # Predicts values only once for each unique attribute in the batch
        # Requires more RAM than the method it overrides
        # Returns [B, N]
        values = torch.empty(attributes.shape[0], self.nentity).to(device=attributes.device)
        # sort beforehand to get indices
        attributes, sort_index = attributes.sort()
        unique_attr, unique_inverse = torch.unique(attributes, sorted=True, return_counts=True)
        values = self.predict_attribute_values(self.ent_embeddings.weight.unsqueeze(1).expand(-1, unique_attr.shape[0], -1), unique_attr)
        # undo unique
        values = torch.repeat_interleave(values, unique_inverse, dim=-1)
        # undo sort
        values = values.index_select(-1, sort_index.argsort(0))
        return values.transpose(0, 1)

    def predict_attribute_values(self, e_emb, attributes):
        # Predict attribute values for a batch
        # e_emb: [B, E], attributes: [B]
        # returns [B]
        # or:
        # e_emb: [B, A, E], attributes: [A]
        # returns [B, A]
        re_head, im_head = torch.chunk(e_emb, 2, dim=-1)
        re_attr, im_attr = torch.chunk(self.attr_embeddings(attributes), 2, dim=-1)
        re_b, im_b = torch.chunk(self.b(attributes), 2, dim=-1)
        re = re_attr * re_head - im_attr * im_head
        im = re_attr * im_head + im_attr * re_head

        re_pred = torch.sum(re, dim=-1) + re_b.view(-1)
        im_pred = torch.sum(im, dim=-1) + im_b.view(-1)
        re_pred = torch.sigmoid(re_pred)
        im_pred = torch.sigmoid(im_pred)

        predictions = (re_pred + im_pred) / 2
        return predictions

    def continuous_loop(self, num_variables, batch_size, atoms, num_var, attr_mask, filters, h_emb_constants, head_vars_mask, conjunction_mask):
        head, rel, tail = atoms[..., 0].long(), atoms[..., 1].long(), atoms[..., 2]
        # var embedding for ID 0 is unused for ease of implementation
        var_embs = nn.Embedding((num_variables * batch_size) + 1, self.rank * 2)
        var_embs.weight.data *= self.init_size

        var_embs.to(atoms.device)
        optimizer = optim.Adam(var_embs.parameters(), lr=0.1)
        prev_loss_value = 1000
        loss_value = 999
        i = 0

        # precompute attribute values for restriction computation
        with torch.no_grad():
            score_restriction_fun = []
            for j in range(num_var):
                if len(attr_mask[:, j].nonzero()) > 0:
                    score_restriction_fun.append(self.score_attribute_restriction(filters[j], rel[:, j]))
                else:
                    score_restriction_fun.append(NotImplementedError)

        # CQD-CO optimization loop
        # Find best embedding for variables simultaniously
        while i < 1000 and math.fabs(prev_loss_value - loss_value) > 1e-9:
            prev_loss_value = loss_value

            # Fill variable positions with optimizable embeddings
            h_emb = h_emb_constants.clone()
            h_emb[head_vars_mask] = var_embs(head[head_vars_mask])

            scores_per_var = []
            factors = []
            for j in range(num_var):
                if len(attr_mask[:, j].nonzero()) > 0:
                    # [B, E] -> [B]
                    score = score_restriction_fun[j](h_emb[:, j, :])
                    scores_per_var.append(score)
                    attr_emb = self.attr_embeddings(rel[:, j])
                    factors.extend([torch.sqrt(attr_emb[..., :self.rank] ** 2 + attr_emb[..., self.rank:] ** 2)])
                    #factors.extend(torch.zeros((score.squeeze().shape[0], self.rank)).to(device=score.device))
                else:
                    r_emb = self.rel_embeddings(rel[:, j])
                    t_emb = var_embs(tail[:, j])
                    score = self.score_o(h_emb[:, j, :], r_emb, t_emb)
                    factors.extend(self.get_factors(*self.split(h_emb[:, j, :], r_emb, t_emb)))
                    scores_per_var.append(score)

            # scores shape: [batch_size, num_var]
            scores = torch.stack(scores_per_var, dim=-1)

            query_score = self.reduce_query_score(scores, conjunction_mask[:, :num_var])

            loss = - query_score.mean() + self.regularizer.forward(factors)
            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1

        return h_emb
