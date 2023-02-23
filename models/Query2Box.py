import torch
import torch.nn as nn
import torch.nn.functional as F

from util import query_name_dict


def Identity(x):
    return x


class BoxOffsetIntersection(nn.Module):

    def __init__(self, dim):
        super(BoxOffsetIntersection, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim)
        self.layer2 = nn.Linear(self.dim, self.dim)

        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate


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


class Query2Box(nn.Module):
    """
    Query2Box with the Attribute Model
    """

    def __init__(self, nentity, nrelation, nattr, use_attributes, hidden_dim,
                 box_mode=('none', 0.2), use_cuda=False):
        super(Query2Box, self).__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.nattr = nattr
        self.use_attributes = use_attributes
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda

        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim

        self.ent_embeddings = nn.Embedding(self.nentity, self.entity_dim)
        self.rel_embeddings = nn.Embedding(self.nrelation, self.relation_dim)
        self.offset_embedding = nn.Embedding(self.nrelation, self.entity_dim)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.offset_embedding.weight.data)

        if self.use_attributes:
            self.attr_embeddings = nn.Embedding(self.nattr, self.entity_dim)
            self.offset_attr_embeddings = nn.Embedding(self.nattr, self.entity_dim)
            nn.init.xavier_uniform_(self.attr_embeddings.weight.data)
            nn.init.xavier_uniform_(self.offset_attr_embeddings.weight.data)

        activation, cen = box_mode
        self.cen = cen  # hyperparameter that balances the in-box distance and the out-box distance
        if activation == 'none':
            self.func = Identity
        elif activation == 'relu':
            self.func = F.relu
        elif activation == 'softplus':
            self.func = F.softplus

        self.center_net = CenterIntersection(self.entity_dim)
        self.offset_net = BoxOffsetIntersection(self.entity_dim)

    def embed_query(self, queries, query_structure, idx):
        '''
        Iterative embed a batch of queries with same structure using Query2box
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
                if self.use_cuda:
                    offset_embedding = torch.zeros_like(embedding).cuda()
                else:
                    offset_embedding = torch.zeros_like(embedding)
                idx += 1
            else:
                embedding, offset_embedding, idx = self.embed_query(queries, query_structure[0], idx)
            for i in range(len(query_structure[-1])):
                r_embedding = self.rel_embeddings(queries[:, idx].long())
                r_offset_embedding = self.offset_embedding(queries[:, idx].long())
                embedding += r_embedding
                offset_embedding += self.func(r_offset_embedding)
                idx += 1
        elif query_structure[1][0] == 'v':
            # ('ap', 'a'), ('v', 'f')
            a = self.attr_embeddings(queries[:, idx+1].long())
            offset_embedding = self.offset_attr_embeddings(queries[:, idx+1].long())
            values = queries[:, idx+2]
            filters = queries[:, idx+3]
            embedding = a
            for i in range(offset_embedding.shape[0]):
                # Scale offset based on the attribute value to filter by
                if filters[i] == -5:
                    offset_embedding[i] *= values[i]
            # intersection with attr_exists likelihood using the additional triples
            attr_exists_queries = (2*queries[:, idx+1].long()+self.nrelation-2*self.nattr).unsqueeze(-1).repeat(1, 2)
            attr_exists_queries[:, 0] = self.nentity-1
            attr_exists_center, attr_exists_offset, _ = self.embed_query(attr_exists_queries, ('e', ('r',)), 0)
            embedding = self.center_net(torch.stack([a, attr_exists_center]))
            offset_embedding = self.offset_net(torch.stack([offset_embedding, attr_exists_offset]))
            idx += 4
        else:
            embedding_list = []
            offset_embedding_list = []
            for i in range(len(query_structure)):
                embedding, offset_embedding, idx = self.embed_query(queries, query_structure[i], idx)
                embedding_list.append(embedding)
                offset_embedding_list.append(offset_embedding)

            embedding = self.center_net(torch.stack(embedding_list))
            offset_embedding = self.offset_net(torch.stack(offset_embedding_list))

        return embedding, offset_embedding, idx

    def transform_union_query(self, queries, query_structure):
        '''
        transform 2u queries to two 1p queries
        transform up queries to two 2p queries
        '''
        if query_name_dict[query_structure] == '2u':
            queries = queries[:, :-1]
        elif query_name_dict[query_structure] == 'up':
            queries = torch.cat([torch.cat([queries[:, :2], queries[:, 5:6]], dim=1), torch.cat([queries[:, 2:4], queries[:, 5:6]], dim=1)], dim=1)
        elif query_name_dict[query_structure] == 'au':
            queries = queries[:, :-1]
        queries = torch.reshape(queries, [queries.shape[0]*2, -1])
        return queries

    def transform_union_structure(self, query_structure):
        if query_name_dict[query_structure] == '2u':
            return ('e', ('r',))
        elif query_name_dict[query_structure] == 'up':
            return ('e', ('r', 'r'))
        elif query_name_dict[query_structure] == 'au':
            return (('ap', 'a'), ('v', 'f'))

    def cal_logit(self, ent_embeddings, query_center_embedding, query_offset_embedding):
        # e.g. [N, E] and [B, 1, E] or [B, nvar, 1, E]
        delta = (ent_embeddings - query_center_embedding).abs()
        distance_out = F.relu(delta - query_offset_embedding)
        distance_in = torch.min(delta, query_offset_embedding)
        # self.cen = \alpha from paper
        dist_box = torch.norm(distance_out, p=1, dim=-1) + self.cen * torch.norm(distance_in, p=1, dim=-1)
        return dist_box

    def score_attr(self, data):
        preds = self.predict_attr(data['batch_e'], data['batch_a'])
        return torch.stack((preds, data['batch_v']), dim=-1)

    def predict_attr(self, entities, attributes, test=False):
        e = self.ent_embeddings(entities)
        a = self.attr_embeddings(attributes)
        offset = self.offset_attr_embeddings(attributes)

        center = a
        delta = (e - center).abs()
        dist = torch.norm(F.relu(offset - delta), p=1, dim=-1)
        offset = torch.norm(offset, p=1, dim=-1)
        return 1 - dist.div(offset)

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

        center_embedding, offset_embedding, _ = self.embed_query(queries, qs, 0)
        positive_logit = self.cal_logit(positive_embedding, center_embedding, offset_embedding)

        if "batch_e" not in data:
            attr_score = torch.FloatTensor(device=positive_embedding.device)
        else:
            attr_score = self.score_attr(data)
        return positive_logit, attr_score

    def forward(self, batch_queries_dict, samples=None):
        """
        If samples is given, compute a score for each query with the given samples as answers. (training)
        If samples is not given, compute a score for each entity to be the answer of a query. (validation/testing)
        """
        if samples is None and self.training:
            raise ValueError("No samples given during training")
        all_scores = list()
        all_center_embeddings, all_offset_embeddings = [], []
        all_union_center_embeddings, all_union_offset_embeddings = [], []

        attr_value_prediction_queries = []
        attr_value_prediction_idxs = []
        for query_structure, queries in batch_queries_dict.items():
            if query_structure[-1][0] == 'ap':
                attr_value_prediction_queries.extend(queries[:, -2:])
                attr_value_prediction_idxs.extend([x + sum([x.shape[0] for x in all_center_embeddings+all_union_center_embeddings]) for x in range(queries.shape[0])])
                queries = queries[:, :-2]
                if query_structure[0] == 'e':
                    embedding = self.ent_embeddings(queries[:, 0].long())
                    all_center_embeddings.append(embedding)
                    all_offset_embeddings.append(torch.zeros_like(embedding))
                    continue
                if len(query_structure) <= 2:
                    query_structure = query_structure[0]
                else:
                    query_structure = query_structure[:-1]
            if 'u' in query_name_dict[query_structure]:
                center_embedding, offset_embedding, _ = self.embed_query(
                    self.transform_union_query(queries, query_structure),
                    self.transform_union_structure(query_structure), 0)
                all_union_center_embeddings.append(center_embedding)
                all_union_offset_embeddings.append(offset_embedding)
            else:
                center_embedding, offset_embedding, _ = self.embed_query(queries, query_structure, 0)
                all_center_embeddings.append(center_embedding)
                all_offset_embeddings.append(offset_embedding)

        if len(all_center_embeddings) > 0 and len(all_offset_embeddings) > 0:
            # [B, 1, E]
            all_center_embeddings = torch.cat(all_center_embeddings, dim=0).unsqueeze(1)
            all_offset_embeddings = torch.cat(all_offset_embeddings, dim=0).unsqueeze(1)
        if len(all_union_center_embeddings) > 0 and len(all_union_offset_embeddings) > 0:
            all_union_center_embeddings = torch.cat(all_union_center_embeddings, dim=0).unsqueeze(1)
            all_union_offset_embeddings = torch.cat(all_union_offset_embeddings, dim=0).unsqueeze(1)
            all_union_center_embeddings = all_union_center_embeddings.view(all_union_center_embeddings.shape[0]//2, 2, 1, -1)
            all_union_offset_embeddings = all_union_offset_embeddings.view(all_union_offset_embeddings.shape[0]//2, 2, 1, -1)

        if len(all_center_embeddings) > 0:
            if self.training:
                embedding = self.ent_embeddings(samples)
                scores = self.cal_logit(embedding, all_center_embeddings, all_offset_embeddings)
            else:
                embedding = self.ent_embeddings.weight
                scores = -self.cal_logit(embedding, all_center_embeddings, all_offset_embeddings)

            if attr_value_prediction_queries:
                delta = (self.ent_embeddings.weight - all_center_embeddings).abs()
                distance_out = torch.norm(F.relu(delta - all_offset_embeddings), p=1, dim=-1)
                zero_dist_out = (distance_out == 0).nonzero(as_tuple=False)

                attributes = torch.stack(attr_value_prediction_queries)[:, -1].long()
                for i, idx in enumerate(attr_value_prediction_idxs):
                    # Predict attribute values for the answers of the query
                    if idx in zero_dist_out[:, 0]:
                        entities = zero_dist_out[zero_dist_out[:, 0] == idx][:, 1]
                        predictions = self.predict_attr(entities, attributes[i].expand_as(entities))
                        scores[idx] = torch.mean(predictions).view(1, 1)
                    else:
                        entities = torch.argmax(scores[idx], dim=-1)
                        scores[idx] = self.predict_attr(entities, attributes[i]).view(1, 1)

            all_scores.extend([s for s in scores])

        if len(all_union_center_embeddings) > 0:
            assert samples is None  # union queries are not used during training
            embedding = self.ent_embeddings.weight
            positive_union_logit = - self.cal_logit(embedding, all_union_center_embeddings, all_union_offset_embeddings)
            positive_union_logit = torch.max(positive_union_logit, dim=1)[0]
            all_scores.extend([s for s in positive_union_logit])

        return all_scores
