import torch
import torch.nn as nn
import torch.nn.functional as F

from util import query_name_dict


class CenterIntersection(nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))  # (num_conj, dim)
        attention = F.softmax(self.layer2(layer1_act), dim=0)  # (num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=0)

        return embedding


class GQE(nn.Module):
    """
    GQE with attribute model. Just for Testing
    """

    def __init__(self, nentity, nrelation, nattr, hidden_dim, gamma, use_cuda=False):
        super(GQE, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.nattr = nattr
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.ent_embeddings = nn.Embedding(self.nentity, self.entity_dim)
        self.rel_embeddings = nn.Embedding(self.nrelation, self.relation_dim)
        self.attr_embeddings = nn.Embedding(self.nattr, self.entity_dim)
        self.b = nn.Embedding(self.nattr, 1)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.attr_embeddings.weight.data)

        self.center_net = CenterIntersection(self.entity_dim)

    def embed_query(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using GQE
        queries: a flattened batch of queries
        '''
        all_relation_flag = True
        for ele in query_structure[-1]:  # whether the current query tree has merged to one branch and only need to do relation traversal, e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding = self.ent_embeddings(queries[:, idx].long())
                idx += 1
            else:
                embedding, idx = self.embed_query(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                if query_structure[-1][i] == 'n':
                    assert False, "vec cannot handle queries with negation"
                else:
                    r_embedding = self.rel_embeddings(queries[:, idx].long())
                    embedding += r_embedding
                idx += 1
        else:
            embedding_list = []
            for i in range(len(query_structure)):
                embedding, idx = self.embed_query(queries, query_structure[i], idx)
                embedding_list.append(embedding)
            embedding = self.center_net(torch.stack(embedding_list))

        return embedding, idx

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if query_name_dict[query_structure] == '2u':
            queries = queries[:, :-1]  # remove union -1
        elif query_name_dict[query_structure] == 'up':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        elif query_name_dict[query_structure] == 'au':
            queries = queries[:, :-1]  # remove union -1
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if query_name_dict[query_structure] == '2u':
            return ('e', ('r',))
        elif query_name_dict[query_structure] == 'up':
            return ('e', ('r', 'r'))
        elif query_name_dict[query_structure] == 'au':
            return (('ap', 'a'), ('v', 'f'))

    def cal_logit(self, entity_embedding, query_embedding):
        distance = entity_embedding - query_embedding
        logit = torch.norm(distance, p=1, dim=-1)
        # logit = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logit

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
        a = self.attr_embeddings(attributes)
        b = self.b(attributes).squeeze()
        predictions = torch.sum(e_emb * a, dim=-1) + b
        return predictions

    def score(self, data):
        """
        For training only.
        """
        h = data['batch_h']
        r = data['batch_r']
        t = data['batch_t']
        qs = ('e', ('r',))

        positive_embedding = self.ent_embeddings(t)
        queries = torch.cat((h.unsqueeze(-1), r.unsqueeze(-1)), dim=-1)

        center_embedding, _ = self.embed_query(queries, qs, 0)
        positive_logit = self.cal_logit(positive_embedding, center_embedding)

        attr_score = self.score_attr(data)
        return positive_logit.squeeze(), attr_score

    def forward(self, batch_queries_dict):
        """
        For evaluation/predictions only.
        """
        all_scores = list()
        all_center_embeddings = []
        all_union_center_embeddings = []

        attr_value_prediction_queries = []
        attr_value_prediction_idxs = []
        for query_structure, queries in batch_queries_dict.items():
            if query_structure[-1][0] == 'ap':
                attr_value_prediction_queries.extend(queries[:, -2:])
                attr_value_prediction_idxs.extend([x + len(all_center_embeddings+all_union_center_embeddings) for x in range(queries.shape[0])])
                queries = queries[:, :-2]
                if len(query_structure) <= 2:
                    query_structure = query_structure[0]
                else:
                    query_structure = query_structure[:-1]
                if query_structure[0] == 'e':
                    #attr_value_prediction_head.extend(queries[:, 0])
                    # continue
                    embedding = self.ent_embeddings(queries[:, 0].long())
                    all_center_embeddings.append(embedding)
                    continue
            if 'u' in query_name_dict[query_structure]:
                center_embedding, _ = self.embed_query(
                    self.transform_union_query(queries, query_structure),
                    self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
            else:
                center_embedding, _ = self.embed_query(queries, query_structure, 0)
                all_center_embeddings.append(center_embedding)

        if len(all_center_embeddings) > 0:
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)

        if len(all_center_embeddings) > 0:
            positive_embedding = self.ent_embeddings.weight
            scores = - self.cal_logit(positive_embedding, all_center_embeddings)
            if attr_value_prediction_queries:
                best_entity = torch.argmax(scores[attr_value_prediction_idxs], dim=-1)
                scores[attr_value_prediction_idxs] = self.predict_attribute_values(best_entity, torch.stack(attr_value_prediction_queries)[:, -1].long()).unsqueeze(1)
            all_scores.extend([s for s in scores])

        if len(all_union_center_embeddings) > 0:
            positive_embedding = self.ent_embeddings.weight
            positive_union_logit = - self.cal_logit(positive_embedding, all_union_center_embeddings)
            positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            all_scores.extend([s for s in positive_union_logit])

        return all_scores
