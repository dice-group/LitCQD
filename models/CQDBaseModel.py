import time
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from models.util import (
    flatten_structure,
    query_to_atoms,
    symbol_placeholder_dict,
    name_query_dict,
    query_name_dict,
)
import models.discrete as d2
import math


class CQDBaseModel(nn.Module):
    MIN_NORM = "min"
    PROD_NORM = "prod"
    NORMS = {MIN_NORM, PROD_NORM}

    def __init__(
        self,
        rank: int,
        nentity: int,
        nrelation: int,
        nattr: int,
        method: str = "discrete",
        t_norm_name: str = "prod",
        query_name_dict: Optional[Dict] = None,
        k: int = 5,
    ):
        super(CQDBaseModel, self).__init__()
        self.rank = rank
        self.nentity = nentity
        self.nrelation = nrelation
        self.nattr = nattr
        self.method = method
        self.t_norm_name = t_norm_name
        self.query_name_dict = query_name_dict
        self.k = k

    def score(self, data):
        raise NotImplementedError()

    def score_o(self, lhs_emb: Tensor, rel_emb: Tensor, rhs_emb: Tensor):
        raise NotImplementedError()

    def score_o_all(self, lhs_emb: Tensor, rel_emb: Tensor, rhs_emb: Tensor):
        raise NotImplementedError()

    def predict_attribute_values_all(self, attributes):
        # Predict attribute values for all entities for a batch of attributes
        # Returns [B, N]
        values = torch.empty(attributes.shape[0], self.nentity).to(
            device=attributes.device
        )
        unique_attr = torch.unique(attributes, sorted=True)
        for i in range(unique_attr.shape[0]):
            idxs = (attributes == unique_attr[i]).nonzero().squeeze(1)
            values[idxs] = self.predict_attribute_values(
                self.ent_embeddings.weight, unique_attr[i].expand(self.nentity)
            ).expand(idxs.shape[0], -1)
        return values

    def predict_attribute_values(self, e_emb, attributes):
        raise NotImplementedError()

    def score_attribute_exists(self, attribute):
        """
        Get a score \in [0,1] representing if an entity has an attribute.
        Based on a relation for each attribute and a dummy entity /attribute/exists:
        (/attribute/exists, r_a, e) for each (e, r_a) if e,r_a \in triples
        """
        
        # using lhs entity and the realtion of the corresponding attribute to do link prediction 
        # o_score_all: compute all the scores of object with the given s, r
        attr_exists_ent = self.ent_embeddings(
            torch.tensor([self.nentity - 1]).to(device=attribute.device)
        )

        a_emb = self.rel_embeddings(
            2 * attribute + self.nrelation - 2 * self.nattr
        )  # *2 to skip inverse relations
        all_scores = self.score_o_all(
            attr_exists_ent.expand(1, 1, -1),
            a_emb.expand(1, 1, -1),
            self.ent_embeddings.weight,
        ).squeeze()

        # ignore score for identity (/attribute/exists)
        all_scores = all_scores[:-1]

        def get_score(e_emb=None):
            """
            Computes scores for the given attribute.
            If e_emb is given, return score for that embedding, otherwise return score for all entities in the dataset.
            """
            if e_emb is not None:
                # Add score for the given e_emb
                e_scores = self.score_o(
                    attr_exists_ent.expand(e_emb.shape[0], -1),
                    a_emb.expand(e_emb.shape[0], -1),
                    e_emb,
                )
                all_scores_tmp = torch.cat((all_scores, e_scores))
            else:
                all_scores_tmp = all_scores

            # Normalize all scores
            all_scores_tmp = self.normalize(all_scores_tmp)

            if e_emb is not None:
                # extract scores for the given e_emb
                scores = all_scores_tmp[-e_emb.shape[0] :]
            else:
                # add score 0 for identity relation to dummy entity
                scores = torch.cat(
                    (
                        all_scores_tmp,
                        torch.FloatTensor([0.0]).to(device=attribute.device),
                    ),
                    dim=-1,
                )
            return scores

        return get_score

    def score_attribute_restriction(self, filters: Tensor, attributes: Tensor):
        """
        filters: [B, 2], attributes: [B], e_emb: [B, E]
        If e_emb is given, return score for these embeddings, otherwise for all entities in the dataset.
        """
        restrictions = filters[..., 1].long()
        values = filters[..., 0]

        all_predictions = self.predict_attribute_values_all(attributes)
        # all_predictions = torch.ones((attributes.shape[0], self.nentity), device=attributes.device)

        # group batch by attriutes
        unique_attr = torch.unique(attributes, sorted=True)

        # precompute scores for all entities
        get_attr_exists_scores = [
            self.score_attribute_exists(attr) for attr in unique_attr
        ]

        def get_stdevs(ids, ids2):
            """
            Compute stdev for the attribute in ids using the predicted values of the entities with ids in ids2.
            ids may contain multiple indices, but they all represent the same attribute. The predicted values are then the same.
            """
            all_preds = all_predictions[ids[0]][ids2]  # [N]
            stdev = torch.std(all_preds, dim=-1, unbiased=False)
            return stdev.expand(1, ids.shape[0])

        def attr_exists_ids_within_stdev(x):
            """
            Return entities most likely having the attribute, based on if their score lies within 1 stdev from the max score.
            """
            res = (x >= torch.max(x) - torch.std(x, dim=-1, unbiased=False)).nonzero(
                as_tuple=True
            )[0]
            if len(res) < 10:
                # Return at least 10 indices for stdev computation
                return x.topk(k=10).indices
            return res

        # stdv of all the values of attributes of each query
        # import numpy as np
        
        # all_values = []
        # for v in self.attr_values.values():
        #   all_values.extend(v)
        
        # stdv = np.std(all_values)
        # stdevs = [torch.as_tensor([stdv], device=attributes.device).expand(1, self.nentity) for i, attr in enumerate(unique_attr)]
  
        # stdv = 0.25
        # if hasattr(self,'attr_values'):
        #   all_values = []
        #   for v in self.attr_values.values():
        #     all_values.extend(v)
        
        #   stdv = np.std(all_values)
        
        
        # calculate the stdv from the values of all the attributes 
        # stdv = self.stdv
        
        
        # precompute standard deviations

        # Use all entities to compute stdev
        stdevs = [
            get_stdevs((attributes == attr).nonzero().squeeze(1), range(self.nentity))
            for i, attr in enumerate(unique_attr)
        ]
        # Use top k entities with highest attr_exists_score to compute stdev
        # stdevs = [get_stdevs((attributes == attr).nonzero().squeeze(1), get_attr_exists_scores[i](None).topk(k=20).indices) for i, attr in enumerate(unique_attr)]
        # Use attributes with max_value - stdev of attr_exists_scores to compute stdev
        # stdevs = [get_stdevs((attributes == attr).nonzero().squeeze(1), attr_exists_ids_within_stdev(get_attr_exists_scores[i](None))) for i, attr in enumerate(unique_attr)]
        # Use a fixed stdev
        # stdevs = [torch.as_tensor([stdv], device=attributes.device).expand(1, self.nentity) for i, attr in enumerate(unique_attr)]

        def score_restriction(restriction, stdev, value, preds):
            """
            Compute a score \in [0,1] indicating if the predictions meet the restriction.
            stdev: [], value: [] or [N], preds: [] or [N]
            Uses CDFs where the mean values are the predicted values.
            """
            # Replace zeros in stdev with ones to allow normal distributions
            
            stdev = stdev.where(stdev != 0, torch.ones_like(stdev))
            factor = 1

            if restriction.item() == symbol_placeholder_dict["="]:
                # TODO: adapt the new equation of the submitted paper
                # print(f'value: {(preds - value).abs()/(factor*stdev)}')
                return 1/torch.exp(((preds - value).abs())/(factor*stdev))
                # return F.relu(1 - (preds - value).abs())
                # print(f'normalized value: {(preds - value).abs()}')
                # print(f'value: {1/torch.exp(10*(((preds - value).abs()/stdev)-0.5))}')
                
                # return 1/(1+torch.exp(10*((preds - value).abs()-0.5)))
              
            elif restriction.item() == symbol_placeholder_dict["<"]:
                # return (preds < value).float()
                # normal = torch.distributions.Normal(preds, stdev)
                # return normal.cdf(value)
                return 1/(1+torch.exp((preds - value)/(factor*stdev)))

            elif restriction.item() == symbol_placeholder_dict[">"]:
                # return (preds > value).float()
                # normal = torch.distributions.Normal(preds, stdev)
                # scores = 1 - normal.cdf(value)
                # if scores.count_nonzero() == 0:
                #     # #     # stdev is that low, that every score is 0
                #     # #     # set to 1 s.t. attr_exists_score is still relevant
                #     scores += 1
                # return scores
                return 1 - 1/(1+torch.exp((preds - value)/(factor*stdev)))
            raise KeyError()

        def _score_attribute_restriction(e_emb=None):
            if e_emb is not None:
                scores = torch.empty((e_emb.shape[0],), requires_grad=True).to(
                    device=e_emb.device
                )
                predictions = self.predict_attribute_values(
                    e_emb, attributes
                ).unsqueeze(-1)
            else:
                scores = torch.empty_like(all_predictions)
                predictions = all_predictions

            for attr_idx, attr in enumerate(unique_attr):
                idxs = (
                    (attributes == attr).nonzero().squeeze(1)
                )  # positions of attr in the batch
                stdev = stdevs[attr_idx]
                if e_emb is None:
                    # stdev is the same for all entities
                    stdev = stdev.expand(self.nentity, -1)

                with torch.no_grad():
                    if e_emb is None:
                        attr_exists_scores = get_attr_exists_scores[attr_idx](None)
                        # scores are the same for each batch entry with that attribute
                        attr_exists_scores = attr_exists_scores.expand(
                            idxs.shape[0], -1
                        )
                    else:
                        attr_exists_scores = get_attr_exists_scores[attr_idx](
                            e_emb[idxs]
                        )

                for i, idx in enumerate(idxs):
                    no_filter_scores = False
                    no_exists_scores = False
                    if no_exists_scores:
                        attr_exists_scores[i] = torch.ones_like(scores[idx])

                    if no_filter_scores:
                        # scores[idx] = torch.ones_like(scores[idx])
                        filter_score = torch.ones_like(scores[idx]) # correct one

                    else:
                        filter_score = score_restriction(
                            restrictions[idx],
                            stdev[..., i],
                            values[idx],
                            predictions[idx],
                        )
                        filter_score = self.normalize(filter_score)
                    # filter_score = score_restriction(restrictions[idx], stdev[..., i], values[idx], predictions[idx])
                    # filter_score = self.normalize(filter_score)
                    # Use minimum:
                    # scores[idx] = torch.where(scores[idx] < attr_exists_scores[i], scores[idx], attr_exists_scores[i])
                    # continue

                    # TODO: change the equation to pass to the submitted paper
                    # if restrictions[idx] in (-5, -6):
                    #     #     # weight attr_exists_scores more if > or < expression
                    #     scores[idx] = torch.sqrt(
                    #         filter_score * attr_exists_scores[i] ** 2
                    #     )

                    # # #     # use mean:
                    # # #     #scores[idx] = (filter_score + attr_exists_scores[i]**2)/2
                    # else:
                    # #     #     # weight restriction scores more if = expression
                    #     scores[idx] = torch.sqrt(
                    #         filter_score**2 * attr_exists_scores[i]
                    #     )
                    # #     # use mean:
                    # #     #scores[idx] = (filter_score**2 + attr_exists_scores[i])/2
                    
                    scores[idx] = filter_score * attr_exists_scores[i]
                    # scores[idx] = torch.pow(filter_score * (attr_exists_scores[i]**4),1/5)
            return scores

        return _score_attribute_restriction

    def batch_t_norm(self, scores: Tensor) -> Tensor:
        if self.t_norm_name == CQDBaseModel.MIN_NORM:
            scores = torch.min(scores, dim=1)[0]
        elif self.t_norm_name == CQDBaseModel.PROD_NORM:
            scores = torch.prod(scores, dim=1)
        else:
            raise ValueError(
                f"t_norm must be one of {CQDBaseModel.NORMS}, got {self.t_norm_name}"
            )

        return scores

    def batch_t_conorm(self, scores: Tensor) -> Tensor:
        if self.t_norm_name == CQDBaseModel.MIN_NORM:
            scores = torch.max(scores, dim=1, keepdim=True)[0]
        elif self.t_norm_name == CQDBaseModel.PROD_NORM:
            scores = torch.sum(scores, dim=1, keepdim=True) - torch.prod(
                scores, dim=1, keepdim=True
            )
        else:
            raise ValueError(
                f"t_norm must be one of {CQDBaseModel.NORMS}, got {self.t_norm_name}"
            )

        return scores

    def normalize_batch(self, tensor):
        # min-max normalize batch
        result = torch.empty_like(tensor)
        for i in range(tensor.shape[0]):
            result[i] = (tensor[i] - tensor[i].min()) / (
                tensor[i].max() - tensor[i].min()
            )
        return result

    def normalize(self, tensor):
        # min-max normalize
        if tensor.shape[0] == 1:
            return tensor
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def reduce_query_score(self, atom_scores, conjunction_mask):
        batch_size, num_atoms, *extra_dims = atom_scores.shape
        atom_scores = torch.sigmoid(atom_scores)
        scores = atom_scores.clone()

        disjunctions = scores[~conjunction_mask].reshape(
            batch_size, -1, *extra_dims
        )  # A or B (union)
        conjunctions = scores[conjunction_mask].reshape(
            batch_size, -1, *extra_dims
        )  # A and B (intersection)

        if disjunctions.shape[1] > 0:
            disjunctions = self.batch_t_conorm(disjunctions)

        conjunctions = torch.cat([disjunctions, conjunctions], dim=1)

        t_norm = self.batch_t_norm(conjunctions)
        return t_norm

    def continuous_loop(
        self,
        num_variables,
        batch_size,
        atoms,
        num_var,
        attr_mask,
        filters,
        h_emb_constants,
        head_vars_mask,
        conjunction_mask,
    ):
        # optimizer and learning rate changed to work with translation-based models
        head, rel, tail = atoms[..., 0].long(), atoms[..., 1].long(), atoms[..., 2]

        # var embedding for ID 0 is unused for ease of implementation
        var_embs = nn.Embedding((num_variables * batch_size) + 1, self.rank * 2)
        nn.init.xavier_uniform_(var_embs.weight.data)

        var_embs.to(atoms.device)
        optimizer = optim.Adagrad(var_embs.parameters(), lr=0.01)
        prev_loss_value = -1000
        loss_value = -999
        i = 0

        # precompute attribute values for restriction computation
        with torch.no_grad():
            score_restriction_fun = []
            for j in range(num_var):
                if len(attr_mask[:, j].nonzero()) > 0:
                    score_restriction_fun.append(
                        self.score_attribute_restriction(filters[j], rel[:, j])
                    )
                else:
                    score_restriction_fun.append(NotImplementedError)

        # CQD-CO optimization loop
        # Find best embedding for variables simultaniously
        while i < 1000 and (
            loss_value > -0.2 or math.fabs(prev_loss_value - loss_value) > 1e-4
        ):
            prev_loss_value = loss_value

            # Fill variable positions with optimizable embeddings
            h_emb = h_emb_constants.clone()
            h_emb[head_vars_mask] = var_embs(head[head_vars_mask])

            scores_per_var = []
            for j in range(num_var):
                if len(attr_mask[:, j].nonzero()) > 0:
                    # [B, 2], [B], [B, E] -> [B] invoked during aip evaluation
                    score = score_restriction_fun[j](h_emb[:, j, :])
                    scores_per_var.append(score.squeeze())
                else:
                    r_emb = self.rel_embeddings(rel[:, j])
                    t_emb = var_embs(tail[:, j])
                    scores_per_var.append(self.score_o(h_emb[:, j, :], r_emb, t_emb))

            # scores shape: [batch_size, num_var]
            scores = torch.stack(scores_per_var, dim=-1)

            query_score = self.reduce_query_score(scores, conjunction_mask[:, :num_var])

            loss = -query_score.mean()
            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1

        return h_emb
  
    def forward(self, batch_queries_dict: Dict[Tuple, Tensor]):
        all_scores = []

        for query_structure, queries in batch_queries_dict.items():
            # print(f"Answering {len(queries)} {query_name_dict[query_structure]} queries...", end='\t', flush=True)
            # start_time = time.time()
            if "continuous" in self.method:
                batch_size = queries.shape[0]
                (
                    atoms,
                    num_variables,
                    conjunction_mask,
                    attr_mask,
                    filters,
                    no_head_entity,
                ) = query_to_atoms(query_structure, queries)

                attr_mask = attr_mask.to(device=queries.device)
                target_mask = torch.sum(atoms == -num_variables, dim=-1) > 0

                # Offsets identify variables across different batches
                var_id_offsets = (
                    torch.arange(batch_size, device=atoms.device) * num_variables
                )
                var_id_offsets = var_id_offsets.reshape(-1, 1, 1)

                # Replace negative variable IDs with valid identifiers
                vars_mask = atoms < 0
                atoms_offset_vars = -atoms + var_id_offsets

                atoms[vars_mask] = atoms_offset_vars[vars_mask]
                head_vars_mask = vars_mask[..., 0]

                head, rel, tail = (
                    atoms[..., 0].long(),
                    atoms[..., 1].long(),
                    atoms[..., 2],
                )

                with torch.no_grad():
                    h_emb = h_emb_constants = self.ent_embeddings(head)

                attr_count = len(attr_mask[0].nonzero())
                filter_count = len([x for x in filters if x is not None])
                attr_prediction = (
                    attr_count > filter_count
                )  # the result is an attribute value

                num_var = atoms.shape[1]
                if attr_count > filter_count:
                    num_var -= 1

                if num_variables > 1:
                    h_emb = self.continuous_loop(
                        num_variables,
                        batch_size,
                        atoms,
                        num_var,
                        attr_mask,
                        filters,
                        h_emb_constants,
                        head_vars_mask,
                        conjunction_mask,
                    )

                with torch.no_grad():
                    # Compute likelihood for each entity in the dataset given h_emb from the CO optimization loop or the head of the query
                    scores = []
                    if len(attr_mask[target_mask].nonzero()) != len(
                        attr_mask[target_mask]
                    ):
                        # using reshape to keep dimensions
                        r_emb = self.rel_embeddings(
                            rel[target_mask & ~attr_mask].reshape(rel.shape[0], -1)
                        )
                        h_emb = h_emb[target_mask & ~attr_mask].reshape(
                            h_emb.shape[0], -1, h_emb.shape[-1]
                        )
                        score = self.score_o_all(
                            h_emb, r_emb, self.ent_embeddings.weight
                        )
                        scores.append(score)

                    # target_mask is equal for each element at dim1 (since query_structure is fixed)
                    filters = [
                        f
                        for i, f in enumerate(filters)
                        if i in target_mask[0].nonzero() and f is not None
                    ]

                    if len(attr_mask[target_mask].nonzero()) > 0:
                        attributes = rel[target_mask & attr_mask]
                        if no_head_entity or len(scores) > 0:
                            # e.g. ai, 2ai, aip, pai, au
                            if filters:
                                predictions = self.score_attribute_restriction(
                                    torch.cat(filters).to(device=attributes.device),
                                    attributes,
                                )(None)
                                predictions = predictions.reshape(
                                    batch_size, -1, self.nentity
                                )
                            else:
                                predictions = self.predict_attribute_values_all(
                                    attributes
                                )
                        else:
                            # e.g. 1ap, 2ap
                            e = h_emb[target_mask & attr_mask]
                            if filters:
                                predictions = self.score_attribute_restriction(
                                    torch.cat(filters).to(device=attributes.device),
                                    attributes,
                                )(e)
                                predictions = predictions.reshape(
                                    batch_size, -1, self.nentity
                                )
                            else:
                                predictions = self.predict_attribute_values(
                                    e, attributes
                                )

                        scores.append(predictions)

                    if not attr_prediction or filters:
                        # intersection / union
                        conjunction_mask = conjunction_mask[target_mask].reshape(
                            conjunction_mask.shape[0], -1
                        )
                        score = torch.cat(scores, dim=1)
                        score = self.reduce_query_score(score, conjunction_mask)
                    else:
                        # attribute value prediction
                        score = scores[0]
                    all_scores.extend([s for s in score])

            elif "discrete" in self.method:
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

                def scoring_function(
                    rel_: Tensor, lhs_: Tensor, rhs_: Tensor
                ) -> Tensor:
                    # answer using desc emb
                    # lhs, rel, _ = self.split(lhs_.squeeze(1), rel_.squeeze(1), rhs_)
                    # pred = self.predict_descriptions(torch.cat(((lhs[0] * rel[0] - lhs[1] * rel[1]), (lhs[1] * rel[0] + lhs[0] * rel[1])), dim=1))
                    # return self.score_description_similarity(pred)
                    return self.score_o_all(lhs_, rel_, rhs_)

                if graph_type == "1p":
                    score = d2.query_1p(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        queries=queries.long(),
                        scoring_function=scoring_function,
                    )
                elif graph_type == "1ap":
                    score = d2.query_1ap(
                        entity_embeddings=self.ent_embeddings,
                        queries=queries.long(),
                        val_prediction_function=self.predict_attribute_values,
                    )
                elif graph_type == "1dp":
                    if hasattr(self, "desc_jointly") and self.desc_jointly:
                        score = d2.query_1dp_jointly(
                            entity_embeddings=self.ent_embeddings,
                            queries=queries.long(),
                            predict_descriptions_function=self.predict_descriptions,
                        )
                    else:
                        score = d2.query_1dp(
                            entity_embeddings=self.ent_embeddings,
                            queries=queries.long(),
                            predict_descriptions_function=self.predict_descriptions,
                            scoring_function=scoring_function,
                        )
                elif graph_type == "2p":
                    score = d2.query_2p(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        queries=queries.long(),
                        scoring_function=scoring_function,
                        k=self.k,
                        t_norm=t_norm,
                    )
                elif graph_type == "2ap":
                    score = d2.query_2ap(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        queries=queries.long(),
                        scoring_function=scoring_function,
                        val_prediction_function=self.predict_attribute_values,
                        k=self.k,
                    )
                elif graph_type == "3p":
                    score = d2.query_3p(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        queries=queries.long(),
                        scoring_function=scoring_function,
                        k=self.k,
                        t_norm=t_norm,
                    )
                elif graph_type == "3ap":
                    score = d2.query_3ap(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        queries=queries.long(),
                        scoring_function=scoring_function,
                        val_prediction_function=self.predict_attribute_values,
                        k=self.k,
                    )
                elif graph_type == "2i":
                    score = d2.query_2i(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        queries=queries.long(),
                        scoring_function=scoring_function,
                        t_norm=t_norm,
                    )
                elif graph_type == "3i":
                    score = d2.query_3i(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        queries=queries.long(),
                        scoring_function=scoring_function,
                        t_norm=t_norm,
                    )
                elif graph_type == "pi":
                    score = d2.query_pi(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        queries=queries.long(),
                        scoring_function=scoring_function,
                        k=self.k,
                        t_norm=t_norm,
                    )
                elif graph_type == "ip":
                    score = d2.query_ip(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        queries=queries.long(),
                        scoring_function=scoring_function,
                        k=self.k,
                        t_norm=t_norm,
                    )
                elif graph_type == "di":
                    if hasattr(self, "desc_jointly") and self.desc_jointly:
                        score = d2.query_di_jointly(
                            entity_embeddings=self.ent_embeddings,
                            queries=queries.long(),
                            predict_descriptions_function=self.predict_descriptions,
                            scoring_function=scoring_function,
                            score_description_similarity=self.score_description_similarity,
                        )
                    else:
                        score = d2.query_di(queries, self.score_description_similarity)
                        # score = d2.query_di(entity_embeddings=self.ent_embeddings,
                        #                    queries=queries,
                        #                    predict_descriptions_function=self.predict_descriptions,
                        #                    scoring_function=scoring_function,
                        #                    score_description_similarity=self.score_description_similarity)
                elif graph_type in ("ai", "ai-lt", "ai-eq", "ai-gt"):
                    score = d2.query_ai(
                        entity_embeddings=self.ent_embeddings,
                        attribute_embeddings=self.attr_embeddings,
                        bias_embeddings=self.b,
                        queries=queries,
                        score_attribute_restriction=self.score_attribute_restriction,
                    )
                elif graph_type == "2ai":
                    score = d2.query_2ai(
                        entity_embeddings=self.ent_embeddings,
                        attribute_embeddings=self.attr_embeddings,
                        bias_embeddings=self.b,
                        queries=queries,
                        score_attribute_restriction=self.score_attribute_restriction,
                        t_norm=t_norm,
                    )
                elif graph_type == "pai":
                    score = d2.query_pai(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        attribute_embeddings=self.attr_embeddings,
                        bias_embeddings=self.b,
                        queries=queries,
                        scoring_function=scoring_function,
                        score_attribute_restriction=self.score_attribute_restriction,
                        t_norm=t_norm,
                        
                    )
                elif graph_type == "aip":
                    score = d2.query_aip(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        attribute_embeddings=self.attr_embeddings,
                        bias_embeddings=self.b,
                        queries=queries,
                        scoring_function=scoring_function,
                        score_attribute_restriction=self.score_attribute_restriction,
                        k=self.k,
                        t_norm=t_norm,
                    )
                elif graph_type == "2u":
                    score = d2.query_2u_dnf(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        queries=queries.long(),
                        scoring_function=scoring_function,
                        t_conorm=t_conorm,
                    )
                elif graph_type == "up":
                    score = d2.query_up_dnf(
                        entity_embeddings=self.ent_embeddings,
                        predicate_embeddings=self.rel_embeddings,
                        queries=queries.long(),
                        scoring_function=scoring_function,
                        k=self.k,
                        t_norm=t_norm,
                        t_conorm=t_conorm,
                    )
                elif graph_type == "au":
                    score = d2.query_au(
                        entity_embeddings=self.ent_embeddings,
                        attribute_embeddings=self.attr_embeddings,
                        bias_embeddings=self.b,
                        queries=queries,
                        score_attribute_restriction=self.score_attribute_restriction,
                        t_conorm=t_conorm,
                    )
                else:
                    raise ValueError(f"Unknown query type: {graph_type}")

                score = [s for s in score]
                all_scores.extend(score)

            # print(" {:2f} ms".format((time.time() - start_time)*1000))

        return all_scores
