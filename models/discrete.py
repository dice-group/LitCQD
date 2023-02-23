import torch
from torch import nn, Tensor

from typing import Callable, Tuple, Optional


def score_candidates(s_emb: Tensor,
                     p_emb: Tensor,
                     candidates_emb: Tensor,
                     k: Optional[int],
                     entity_embeddings: nn.Module,
                     scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Tuple[Tensor, Optional[Tensor]]:

    batch_size = max(s_emb.shape[0], p_emb.shape[0])
    embedding_size = s_emb.shape[1]

    def reshape(emb: Tensor) -> Tensor:
        if emb.shape[0] < batch_size:
            n_copies = batch_size // emb.shape[0]
            emb = emb.reshape(-1, 1, embedding_size).repeat(1, n_copies, 1).reshape(-1, embedding_size)
        return emb

    s_emb = reshape(s_emb).unsqueeze(1)
    p_emb = reshape(p_emb).unsqueeze(1)
    nb_entities = candidates_emb.shape[0]

    x_k_emb_3d = None

    # [B, N]
    atom_scores_2d = scoring_function(s_emb, p_emb, candidates_emb).squeeze(1)
    atom_k_scores_2d = atom_scores_2d

    if k is not None:
        k_ = min(k, nb_entities)

        # [B, K], [B, K]
        atom_k_scores_2d, atom_k_indices = torch.topk(atom_scores_2d, k=k_, dim=1, largest=True)

        # [B, K, E]
        x_k_emb_3d = entity_embeddings(atom_k_indices)

    return atom_k_scores_2d, x_k_emb_3d


def query_1p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Tensor:
    s_emb = entity_embeddings(queries[:, 0])
    p_emb = predicate_embeddings(queries[:, 1])
    candidates_emb = entity_embeddings.weight

    assert queries.shape[1] == 2

    res, _ = score_candidates(s_emb=s_emb, p_emb=p_emb,
                              candidates_emb=candidates_emb, k=None,
                              entity_embeddings=entity_embeddings,
                              scoring_function=scoring_function)

    return res


def query_1ap(entity_embeddings: nn.Module,
              queries: Tensor,
              val_prediction_function: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    return val_prediction_function(s_emb, queries[:, 2]).unsqueeze(-1)


def query_1dp_jointly(entity_embeddings: nn.Module,
                      queries: Tensor,
                      predict_descriptions_function,
                      ) -> Tensor:

    return predict_descriptions_function(entity_embeddings(queries[:, 0]))


def query_1dp(entity_embeddings: nn.Module,
              queries: Tensor,
              predict_descriptions_function,
              scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              ) -> Tensor:

    e_emb = entity_embeddings(queries[:, 0])
    return predict_descriptions_function(e_emb)
    # take k nearest neighbours into account
    batch_size = queries.shape[0]
    k = 10

    rel_emb = torch.ones_like(e_emb)
    # _, [B, K, E]
    _, nearest_neighbours = score_candidates(
        s_emb=e_emb,
        p_emb=rel_emb,
        candidates_emb=entity_embeddings.weight,
        k=k,
        entity_embeddings=entity_embeddings,
        scoring_function=scoring_function,
    )

    desc_emb_pred = predict_descriptions_function(nearest_neighbours.view(batch_size * k, -1))
    desc_emb_pred = desc_emb_pred.view(batch_size, k, -1)

    desc_emb_pred_mean = torch.mean(desc_emb_pred, dim=1)
    # desc_emb_pred_mean = desc_emb_pred[:, 1, :]
    return desc_emb_pred_mean


def query_2p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]

    # [B, K], [B, K, E]
    atom1_k_scores_2d, x1_k_emb_3d = score_candidates(s_emb=s_emb, p_emb=p1_emb,
                                                      candidates_emb=candidates_emb, k=k,
                                                      entity_embeddings=entity_embeddings,
                                                      scoring_function=scoring_function)

    # [B * K, E]
    x1_k_emb_2d = x1_k_emb_3d.reshape(-1, emb_size)

    # [B * K, N]
    atom2_scores_2d, _ = score_candidates(s_emb=x1_k_emb_2d, p_emb=p2_emb,
                                          candidates_emb=candidates_emb, k=None,
                                          entity_embeddings=entity_embeddings,
                                          scoring_function=scoring_function)

    # [B, K] -> [B, K, N]
    atom1_scores_3d = atom1_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)
    # [B * K, N] -> [B, K, N]
    atom2_scores_3d = atom2_scores_2d.reshape(batch_size, -1, nb_entities)

    res = t_norm(atom1_scores_3d, atom2_scores_3d)

    # [B, K, N] -> [B, N]
    res, _ = torch.max(res, dim=1)
    return res


def query_2ap(entity_embeddings: nn.Module,
              predicate_embeddings: nn.Module,
              queries: Tensor,
              scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              val_prediction_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              k: int) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    p_emb = predicate_embeddings(queries[:, 1])
    candidates_emb = entity_embeddings.weight

    # [B, K, E]
    _, res = score_candidates(s_emb=s_emb, p_emb=p_emb,
                              candidates_emb=candidates_emb, k=k,
                              entity_embeddings=entity_embeddings,
                              scoring_function=scoring_function)

    # [K, B, E]
    res = res.transpose(0, 1)

    # [K * B, E]
    res = res.reshape(-1, res.shape[-1])

    # [K * B]
    predictions = val_prediction_function(res, queries[:, 3].repeat(k))
    # [K, B]
    predictions = predictions.reshape(k, -1)

    # [B, K]
    predictions = predictions.transpose(0, 1)
    return predictions


def query_3p(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])
    p3_emb = predicate_embeddings(queries[:, 3])

    candidates_emb = entity_embeddings.weight
    nb_entities = candidates_emb.shape[0]

    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]

    # [B, K], [B, K, E]
    atom1_k_scores_2d, x1_k_emb_3d = score_candidates(s_emb=s_emb, p_emb=p1_emb,
                                                      candidates_emb=candidates_emb, k=k,
                                                      entity_embeddings=entity_embeddings,
                                                      scoring_function=scoring_function)

    # [B * K, E]
    x1_k_emb_2d = x1_k_emb_3d.reshape(-1, emb_size)

    # [B * K, K], [B * K, K, E]
    atom2_k_scores_2d, x2_k_emb_3d = score_candidates(s_emb=x1_k_emb_2d, p_emb=p2_emb,
                                                      candidates_emb=candidates_emb, k=k,
                                                      entity_embeddings=entity_embeddings,
                                                      scoring_function=scoring_function)

    # [B * K * K, E]
    x2_k_emb_2d = x2_k_emb_3d.reshape(-1, emb_size)

    # [B * K * K, N]
    atom3_scores_2d, _ = score_candidates(s_emb=x2_k_emb_2d, p_emb=p3_emb,
                                          candidates_emb=candidates_emb, k=None,
                                          entity_embeddings=entity_embeddings,
                                          scoring_function=scoring_function)

    # [B, K] -> [B, K, N]
    atom1_scores_3d = atom1_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)

    # [B * K, K] -> [B, K * K, N]
    atom2_scores_3d = atom2_k_scores_2d.reshape(batch_size, -1, 1).repeat(1, 1, nb_entities)

    # [B * K * K, N] -> [B, K * K, N]
    atom3_scores_3d = atom3_scores_2d.reshape(batch_size, -1, nb_entities)

    # [B, K, N] -> [B, K * K, N]
    atom1_scores_3d = atom1_scores_3d.repeat(1, atom3_scores_3d.shape[1] // atom1_scores_3d.shape[1], 1)

    res = t_norm(atom1_scores_3d, atom2_scores_3d)
    res = t_norm(res, atom3_scores_3d)

    # [B, K, N] -> [B, N]
    res, _ = torch.max(res, dim=1)
    return res


def query_3ap(entity_embeddings: nn.Module,
              predicate_embeddings: nn.Module,
              queries: Tensor,
              scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              val_prediction_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              k: int) -> Tensor:

    s_emb = entity_embeddings(queries[:, 0])
    p1_emb = predicate_embeddings(queries[:, 1])
    p2_emb = predicate_embeddings(queries[:, 2])
    candidates_emb = entity_embeddings.weight
    batch_size = s_emb.shape[0]
    emb_size = s_emb.shape[1]

    # [B, K, E]
    _, res1 = score_candidates(s_emb=s_emb, p_emb=p1_emb,
                               candidates_emb=candidates_emb, k=k,
                               entity_embeddings=entity_embeddings,
                               scoring_function=scoring_function)

    # [B * K, K, E]
    _, res2 = score_candidates(s_emb=res1.reshape(-1, emb_size), p_emb=p2_emb,
                               candidates_emb=candidates_emb, k=k,
                               entity_embeddings=entity_embeddings,
                               scoring_function=scoring_function)

    # [B, K * K, E]
    res2 = res2.reshape(batch_size, -1, emb_size)
    # [K * K, B, E]
    res2 = res2.transpose(0, 1)

    # [K * K * B, E]
    res2 = res2.reshape(-1, res2.shape[-1])

    # [K * K * B]
    predictions = val_prediction_function(res2, queries[:, 4].repeat(k*k))
    # [K * K, B]
    predictions = predictions.reshape(k*k, batch_size)

    # [B, K * K]
    predictions = predictions.transpose(0, 1)
    return predictions


def query_2i(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function)

    res = t_norm(scores_1, scores_2)

    return res


def query_3i(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function)
    scores_3 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 4:6], scoring_function=scoring_function)

    res = t_norm(scores_1, scores_2)
    res = t_norm(res, scores_3)

    return res


def query_ip(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    # [B, N]
    scores_1 = query_2i(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:4], scoring_function=scoring_function, t_norm=t_norm)

    # [B, E]
    p_emb = predicate_embeddings(queries[:, 4])

    batch_size = p_emb.shape[0]
    emb_size = p_emb.shape[1]

    # [N, E]
    e_emb = entity_embeddings.weight
    nb_entities = e_emb.shape[0]

    k_ = min(k, nb_entities)

    # [B, K], [B, K]
    scores_1_k, scores_1_k_indices = torch.topk(scores_1, k=k_, dim=1)

    # [B, K, E]
    scores_1_k_emb = entity_embeddings(scores_1_k_indices)

    # [B * K, E]
    scores_1_k_emb_2d = scores_1_k_emb.reshape(batch_size * k_, emb_size)

    # [B * K, N]
    scores_2, _ = score_candidates(s_emb=scores_1_k_emb_2d, p_emb=p_emb, candidates_emb=e_emb, k=None,
                                   entity_embeddings=entity_embeddings, scoring_function=scoring_function)

    # [B * K, N]
    scores_1_k = scores_1_k.reshape(batch_size, k_, 1).repeat(1, 1, nb_entities)
    scores_2 = scores_2.reshape(batch_size, k_, nb_entities)

    res = t_norm(scores_1_k, scores_2)
    res, _ = torch.max(res, dim=1)

    return res


def query_pi(entity_embeddings: nn.Module,
             predicate_embeddings: nn.Module,
             queries: Tensor,
             scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
             k: int,
             t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_2p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:3], scoring_function=scoring_function, k=k, t_norm=t_norm)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 3:5], scoring_function=scoring_function)

    res = t_norm(scores_1, scores_2)

    return res


def query_di(queries: Tensor,
             score_description_similarity) -> Tensor:
    return score_description_similarity(queries[:, 1:-1])


def query_di2(entity_embeddings: nn.Module,
              queries: Tensor,
              predict_descriptions_function,
              scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              score_description_similarity) -> Tensor:
    # return score_description_similarity(queries[:, 1:-1])
    expected = queries[:, 1:-1]
    queries = torch.as_tensor([[e, -7] for e in range(entity_embeddings.weight.shape[0])], device=entity_embeddings.weight.device)
    prediction = query_1dp(entity_embeddings, queries, predict_descriptions_function, scoring_function)

    vec_norm = torch.linalg.norm(expected, dim=1, keepdim=True)  # [batch_size, 1]
    pred_norm = torch.linalg.norm(prediction.T, dim=0, keepdim=True)  # [1, nentity]

    cos_sim = ((expected @ prediction.T) / (vec_norm @ pred_norm))
    return cos_sim


def query_di_jointly(entity_embeddings: nn.Module,
                     queries: Tensor,
                     predict_descriptions_function,
                     scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                     score_description_similarity) -> Tensor:

    res = score_description_similarity(queries[:, 1:-1])
    # res = torch.prod(res, dim=1)
    # res = torch.mean(res, dim=1)
    return res
    expected = queries[:, 1:-1]
    queries = torch.as_tensor([[e, -7] for e in range(entity_embeddings.weight.shape[0])], device=entity_embeddings.weight.device)
    prediction = query_1dp(entity_embeddings, queries, predict_descriptions_function, scoring_function)

    vec_norm = torch.linalg.norm(expected, dim=1, keepdim=True)  # [batch_size, 1]
    pred_norm = torch.linalg.norm(prediction.T, dim=0, keepdim=True)  # [1, nentity]

    cos_sim = ((expected @ prediction.T) / (vec_norm @ pred_norm))
    return cos_sim


def query_ai(entity_embeddings: nn.Module,
             attribute_embeddings: nn.Module,
             bias_embeddings: nn.Module,
             queries: Tensor,
             score_attribute_restriction: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    # [B], [B] -> [B, 2]
    filters = torch.stack((queries[:, 2], queries[:, 3].long()), dim=-1)

    return score_attribute_restriction(filters, queries[:, 1].long())(None)

    s_emb = entity_embeddings.weight.unsqueeze(1).expand(-1, queries.shape[0], -1)
    a_emb = attribute_embeddings(queries[:, 1])
    b_emb = bias_embeddings(queries[:, 1])

    # evaluate each query separately to avoid running out of memory
    predictions_per_query = []
    for i in range(queries.shape[0]):
        # [N]
        predictions_per_query.append(val_prediction_function(s_emb[:, i, :], a_emb[i], b_emb[i]))

    # [N, B]
    predictions = torch.stack(predictions_per_query, dim=-1)
    # [B, N]
    predictions = predictions.transpose(0, 1)

    # [B], [B] -> [B, 2]
    filters = torch.stack((queries[:, 2], queries[:, 3]), dim=-1)
    # [B, N]
    scores = score_attribute_restriction(predictions, filters)
    return scores


def query_2ai(entity_embeddings: nn.Module,
              attribute_embeddings: nn.Module,
              bias_embeddings: nn.Module,
              queries: Tensor,
              score_attribute_restriction: Callable[[Tensor, Tensor], Tensor],
              t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_ai(entity_embeddings, attribute_embeddings, bias_embeddings, queries[:, 0:4], score_attribute_restriction)
    scores_2 = query_ai(entity_embeddings, attribute_embeddings, bias_embeddings, queries[:, 4:8], score_attribute_restriction)

    res = t_norm(scores_1, scores_2)
    return res


def query_pai(entity_embeddings: nn.Module,
              predicate_embeddings: nn.Module,
              attribute_embeddings: nn.Module,
              bias_embeddings: nn.Module,
              queries: Tensor,
              scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              score_attribute_restriction: Callable[[Tensor, Tensor], Tensor],
              t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    """
    1. Get the attr value for k entities only (using 2ap); hits@i metric working? What if k < #answers?
    2. Predict value for all entities.
    For aip:
    1. Predict attr value for k entities only (using 2ap) and rank all entities based on |their value - mean value of k entities|?? nah, no restriction
    Both do not use t_norm. It is simple link prediction only. How could a more complex query look like?

    Ended up doing an intersection of result of 1p and result of ai.
    """
    # [B, N]
    scores_1 = query_1p(entity_embeddings, predicate_embeddings, queries[:, :2].long(), scoring_function)
    # [B, N]
    scores_2 = query_ai(entity_embeddings, attribute_embeddings, bias_embeddings, queries[:, 2:], score_attribute_restriction)
    scores = t_norm(scores_1, scores_2)
    return scores


def query_aip(entity_embeddings: nn.Module,
              predicate_embeddings: nn.Module,
              attribute_embeddings: nn.Module,
              bias_embeddings: nn.Module,
              queries: Tensor,
              scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
              score_attribute_restriction: Callable[[Tensor, Tensor], Tensor],
              k: int,
              t_norm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    p_emb = predicate_embeddings(queries[:, -1].long())
    batch_size = p_emb.shape[0]
    emb_size = p_emb.shape[1]

    # [B, N]
    scores_1 = query_ai(entity_embeddings, attribute_embeddings, bias_embeddings, queries[:, :4], score_attribute_restriction)

    # [N, E]
    e_emb = entity_embeddings.weight
    nb_entities = e_emb.shape[0]

    k_ = min(k, nb_entities)

    # [B, K], [B, K]
    scores_1_k, scores_1_k_indices = torch.topk(scores_1, k=k_, dim=1)

    # [B, K, E]
    scores_1_k_emb = entity_embeddings(scores_1_k_indices)

    # [B * K, E]
    scores_1_k_emb_2d = scores_1_k_emb.reshape(batch_size * k_, emb_size)

    # [B * K, N]
    scores_2, _ = score_candidates(s_emb=scores_1_k_emb_2d, p_emb=p_emb, candidates_emb=e_emb, k=None,
                                   entity_embeddings=entity_embeddings, scoring_function=scoring_function)

    # [B * K, N]
    scores_1_k = scores_1_k.reshape(batch_size, k_, 1).repeat(1, 1, nb_entities)
    scores_2 = scores_2.reshape(batch_size, k_, nb_entities)

    res = t_norm(scores_1_k, scores_2)
    res, _ = torch.max(res, dim=1)
    return res


def query_2u_dnf(entity_embeddings: nn.Module,
                 predicate_embeddings: nn.Module,
                 queries: Tensor,
                 scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                 t_conorm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 0:2], scoring_function=scoring_function)
    scores_2 = query_1p(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                        queries=queries[:, 2:4], scoring_function=scoring_function)

    res = t_conorm(scores_1, scores_2)

    return res


def query_up_dnf(entity_embeddings: nn.Module,
                 predicate_embeddings: nn.Module,
                 queries: Tensor,
                 scoring_function: Callable[[Tensor, Tensor, Tensor], Tensor],
                 k: int,
                 t_norm: Callable[[Tensor, Tensor], Tensor],
                 t_conorm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    # [B, N]
    scores_1 = query_2u_dnf(entity_embeddings=entity_embeddings, predicate_embeddings=predicate_embeddings,
                            queries=queries[:, 0:4], scoring_function=scoring_function, t_conorm=t_conorm)

    # [B, E]
    p_emb = predicate_embeddings(queries[:, 5])

    batch_size = p_emb.shape[0]
    emb_size = p_emb.shape[1]

    # [N, E]
    e_emb = entity_embeddings.weight
    nb_entities = e_emb.shape[0]

    k_ = min(k, nb_entities)

    # [B, K], [B, K]
    scores_1_k, scores_1_k_indices = torch.topk(scores_1, k=k_, dim=1)

    # [B, K, E]
    scores_1_k_emb = entity_embeddings(scores_1_k_indices)

    # [B * K, E]
    scores_1_k_emb_2d = scores_1_k_emb.reshape(batch_size * k_, emb_size)

    # [B * K, N]
    scores_2, _ = score_candidates(s_emb=scores_1_k_emb_2d, p_emb=p_emb, candidates_emb=e_emb, k=None,
                                   entity_embeddings=entity_embeddings, scoring_function=scoring_function)

    # [B * K, N]
    scores_1_k = scores_1_k.reshape(batch_size, k_, 1).repeat(1, 1, nb_entities)
    scores_2 = scores_2.reshape(batch_size, k_, nb_entities)

    res = t_norm(scores_1_k, scores_2)
    res, _ = torch.max(res, dim=1)

    return res


def query_au(entity_embeddings: nn.Module,
             attribute_embeddings: nn.Module,
             bias_embeddings: nn.Module,
             queries: Tensor,
             score_attribute_restriction: Callable[[Tensor, Tensor], Tensor],
             t_conorm: Callable[[Tensor, Tensor], Tensor]) -> Tensor:

    scores_1 = query_ai(entity_embeddings, attribute_embeddings, bias_embeddings, queries[:, 0:4], score_attribute_restriction)
    scores_2 = query_ai(entity_embeddings, attribute_embeddings, bias_embeddings, queries[:, 4:8], score_attribute_restriction)

    res = t_conorm(scores_1, scores_2)
    return res
