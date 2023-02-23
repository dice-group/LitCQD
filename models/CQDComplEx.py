"""
Used to evaluate the published official CQD model. Not relevant for the thesis.
"""
from models.CQDBaseModel import CQDBaseModel

import torch
import torch.nn as nn
from torch import optim, Tensor
import math

from models.util import query_to_atoms
import models.discrete as d2

from typing import Tuple, List, Optional, Dict


class N3:
    def __init__(self, weight: float):
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class CQDComplEx(CQDBaseModel):

    def __init__(self,
                 init_size: float = 1e-3,
                 reg_weight: float = 1e-2,
                 do_sigmoid: bool = False,
                 do_normalize: bool = False,
                 use_cuda: bool = False,
                 *args,
                 **kwargs,
                 ):
        super(CQDComplEx, self).__init__(*args, **kwargs)

        sizes = (self.nentity, self.nrelation)
        self.embeddings = nn.ModuleList([nn.Embedding(s, 2 * self.rank, sparse=True) for s in sizes[:2]])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.init_size = init_size
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.regularizer = N3(reg_weight)

        self.do_sigmoid = do_sigmoid
        self.do_normalize = do_normalize

        self.use_cuda = use_cuda

    def split(self,
              lhs_emb: Tensor,
              rel_emb: Tensor,
              rhs_emb: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        lhs = lhs_emb[..., :self.rank], lhs_emb[..., self.rank:]
        rel = rel_emb[..., :self.rank], rel_emb[..., self.rank:]
        rhs = rhs_emb[..., :self.rank], rhs_emb[..., self.rank:]
        return lhs, rel, rhs

    def loss(self, data, attr_loss_fn) -> Tensor:
        triples = torch.cat((data['batch_h'].unsqueeze(1), data['batch_r'].unsqueeze(1), data['batch_t'].unsqueeze(1)), -1)
        (scores_o, scores_s), factors = self.score_candidates(triples)
        l_fit = self.loss_fn(scores_o, triples[:, 2]) + self.loss_fn(scores_s, triples[:, 0])
        l_reg = self.regularizer.forward(factors)
        return l_fit + l_reg

    def score_candidates(self, triples: Tensor) -> Tuple[Tuple[Tensor, Tensor], Optional[List[Tensor]]]:
        lhs_emb = self.embeddings[0](triples[:, 0])
        rel_emb = self.embeddings[1](triples[:, 1])
        rhs_emb = self.embeddings[0](triples[:, 2])
        to_score = self.embeddings[0].weight
        scores_o, _ = self.score_o(lhs_emb, rel_emb, to_score)
        scores_s, _ = self.score_s(to_score, rel_emb, rhs_emb)
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        factors = self.get_factors(lhs, rel, rhs)
        return (scores_o, scores_s), factors

    def score_o(self,
                lhs_emb: Tensor,
                rel_emb: Tensor,
                rhs_emb: Tensor,
                return_factors: bool = False) -> Tuple[Tensor, Optional[List[Tensor]]]:
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        score_1 = (lhs[0] * rel[0] - lhs[1] * rel[1]) @ rhs[0].transpose(-1, -2)
        score_2 = (lhs[1] * rel[0] + lhs[0] * rel[1]) @ rhs[1].transpose(-1, -2)
        factors = self.get_factors(lhs, rel, rhs) if return_factors else None
        return score_1 + score_2, factors

    def score_s(self,
                lhs_emb: Tensor,
                rel_emb: Tensor,
                rhs_emb: Tensor,
                return_factors: bool = False) -> Tuple[Tensor, Optional[List[Tensor]]]:
        lhs, rel, rhs = self.split(lhs_emb, rel_emb, rhs_emb)
        score_1 = (rhs[0] * rel[0] + rhs[1] * rel[1]) @ lhs[0].transpose(-1, -2)
        score_2 = (rhs[1] * rel[0] - rhs[0] * rel[1]) @ lhs[1].transpose(-1, -2)
        factors = self.get_factors(lhs, rel, rhs) if return_factors else None
        return score_1 + score_2, factors

    def get_factors(self,
                    lhs: Tuple[Tensor, Tensor],
                    rel: Tuple[Tensor, Tensor],
                    rhs: Tuple[Tensor, Tensor]) -> List[Tensor]:
        factors = []
        for term in (lhs, rel, rhs):
            factors.append(torch.sqrt(term[0] ** 2 + term[1] ** 2))
        return factors

    def forward(self, batch_queries_dict: Dict[Tuple, Tensor]):
        all_scores = []

        for query_structure, queries in batch_queries_dict.items():
            batch_size = queries.shape[0]

            atoms, num_variables, conjunction_mask, _, _, _ = query_to_atoms(query_structure, queries)

            target_mask = torch.sum(atoms == -num_variables, dim=-1) > 0

            # Offsets identify variables across different batches
            var_id_offsets = torch.arange(batch_size, device=atoms.device) * num_variables
            var_id_offsets = var_id_offsets.reshape(-1, 1, 1)

            # Replace negative variable IDs with valid identifiers
            vars_mask = atoms < 0
            atoms_offset_vars = -atoms + var_id_offsets

            atoms[vars_mask] = atoms_offset_vars[vars_mask]
            head_vars_mask = vars_mask[..., 0]

            head, rel, tail = atoms[..., 0].long(), atoms[..., 1].long(), atoms[..., 2]

            with torch.no_grad():
                h_emb = h_emb_constants = self.embeddings[0](head)
                r_emb = self.embeddings[1](rel)

            if 'continuous' in self.method:
                if num_variables > 1:
                    # var embedding for ID 0 is unused for ease of implementation
                    var_embs = nn.Embedding((num_variables * batch_size) + 1, self.rank * 2)
                    var_embs.weight.data *= self.init_size
                    # nn.init.xavier_uniform_(var_embs.weight.data)

                    var_embs.to(atoms.device)
                    optimizer = optim.Adam(var_embs.parameters(), lr=0.1)
                    #optimizer = optim.SGD(var_embs.parameters(), lr=1.0)
                    prev_loss_value = 1000
                    loss_value = 999
                    i = 0

                    # CQD-CO optimization loop
                    # Find best embedding for variables simultaniously
                    while i < 1000 and math.fabs(prev_loss_value - loss_value) > 1e-9:
                        prev_loss_value = loss_value

                        h_emb = h_emb_constants.clone()
                        # Fill variable positions with optimizable embeddings
                        h_emb[head_vars_mask] = var_embs(head[head_vars_mask])
                        t_emb = var_embs(tail)
                        scores, factors = self.score_o(h_emb.unsqueeze(-2),
                                                       r_emb.unsqueeze(-2),
                                                       t_emb.unsqueeze(-2),
                                                       return_factors=True)
                        query_score = self.reduce_query_score(scores, conjunction_mask)

                        loss = - query_score.mean() + self.regularizer.forward(factors)
                        loss_value = loss.item()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        i += 1

                with torch.no_grad():
                    # Select predicates involving target variable only
                    conjunction_mask = conjunction_mask[target_mask].reshape(batch_size, -1)

                    target_mask = target_mask.unsqueeze(-1).expand_as(h_emb)
                    emb_size = h_emb.shape[-1]
                    h_emb = h_emb[target_mask].reshape(batch_size, -1, emb_size)
                    r_emb = r_emb[target_mask].reshape(batch_size, -1, emb_size)
                    to_score = self.embeddings[0].weight

                    scores, _ = self.score_o(h_emb, r_emb, to_score)
                    query_score = self.reduce_query_score(scores, conjunction_mask)
                    all_scores.extend([s for s in query_score])

            elif 'discrete' in self.method:
                queries = queries.long()
                graph_type = self.query_name_dict[query_structure]

                def t_norm(a: Tensor, b: Tensor) -> Tensor:
                    return torch.minimum(a, b)

                def t_conorm(a: Tensor, b: Tensor) -> Tensor:
                    return torch.maximum(a, b)

                if self.t_norm_name == CQDBaseModel.PROD_NORM:
                    def t_norm(a: Tensor, b: Tensor) -> Tensor:
                        return a * b

                    def t_conorm(a: Tensor, b: Tensor) -> Tensor:
                        return 1 - ((1 - a) * (1 - b))

                def normalize(scores_: Tensor) -> Tensor:
                    scores_ = scores_ - scores_.min(1, keepdim=True)[0]
                    scores_ = scores_ / scores_.max(1, keepdim=True)[0]
                    return scores_

                def scoring_function(rel_: Tensor, lhs_: Tensor, rhs_: Tensor) -> Tensor:
                    res, _ = self.score_o(lhs_, rel_, rhs_)
                    if self.do_sigmoid is True:
                        res = torch.sigmoid(res)
                    if self.do_normalize is True:
                        res = normalize(res)
                    return res

                if graph_type == "1p":
                    scores = d2.query_1p(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function)
                elif graph_type == "2p":
                    scores = d2.query_2p(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "3p":
                    scores = d2.query_3p(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "2i":
                    scores = d2.query_2i(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function, t_norm=t_norm)
                elif graph_type == "3i":
                    scores = d2.query_3i(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function, t_norm=t_norm)
                elif graph_type == "pi":
                    scores = d2.query_pi(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "ip":
                    scores = d2.query_ip(entity_embeddings=self.embeddings[0],
                                         predicate_embeddings=self.embeddings[1],
                                         queries=queries,
                                         scoring_function=scoring_function,
                                         k=self.k, t_norm=t_norm)
                elif graph_type == "2u":
                    scores = d2.query_2u_dnf(entity_embeddings=self.embeddings[0],
                                             predicate_embeddings=self.embeddings[1],
                                             queries=queries,
                                             scoring_function=scoring_function,
                                             t_conorm=t_conorm)
                elif graph_type == "up":
                    scores = d2.query_up_dnf(entity_embeddings=self.embeddings[0],
                                             predicate_embeddings=self.embeddings[1],
                                             queries=queries,
                                             scoring_function=scoring_function,
                                             k=self.k, t_norm=t_norm, t_conorm=t_conorm)
                else:
                    raise ValueError(f'Unknown query type: {graph_type}')

                all_scores.extend([s for s in scores])

        return all_scores
