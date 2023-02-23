from bidict import bidict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


import gensim
import gensim.downloader

from models import CQDComplExD


class CQDComplExDJointly(CQDComplExD):
    """
    Testing joint learning of word embeddings and entity/relation embeddings. Only for Testing
    """

    def __init__(self,
                 descriptions: dict,
                 word2id: bidict,
                 word_emb_dim: int = 100,
                 init_size: float = 1e-3,
                 *args,
                 **kwargs,
                 ):
        word_embeddings_dim = word_emb_dim
        self.rank = word_embeddings_dim // 2

        super(CQDComplExDJointly, self).__init__(*args, **kwargs)

        vocab_size = len(word2id)
        self.desc_jointly = True

        load_pretrained = False
        load_selftrained = False
        if load_pretrained:
            model = gensim.downloader.load('word2vec-google-news-300')
            pretrained = list()
            for i in range(vocab_size):
                word = word2id.inv[i]
                try:
                    pretrained.append(model[word])
                except KeyError:
                    pretrained.append([init_size]*word_embeddings_dim)

            self.word_embeddings = nn.Embedding.from_pretrained(torch.as_tensor(pretrained), freeze=False)
        elif load_selftrained:
            model = gensim.models.Word2Vec.load('data/scripts/data/textual/LiteralE/word2vec_model').wv
            pretrained = list()
            for i in range(vocab_size):
                word = word2id.inv[i]
                try:
                    pretrained.append(model[word])
                except KeyError:
                    pretrained.append([init_size]*word_embeddings_dim)

            self.word_embeddings = nn.Embedding.from_pretrained(torch.as_tensor(pretrained), freeze=True)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, word_embeddings_dim)
            self.word_embeddings.weight.data *= init_size

        self.keywords = descriptions
        for i in range(self.nentity):
            if i not in self.keywords:
                self.keywords[i] = False
            else:
                self.keywords[i] = torch.as_tensor(self.keywords[i], device=torch.device("cuda:0"))

    def get_description_embedding(self, ent):
        words = list()
        for e in ent:
            keywords = self.keywords[e.item()]
            if type(keywords) == bool and keywords == False:
                words.append(torch.ones_like(self.word_embeddings.weight[0]))
            else:
                words.append(torch.mean(self.word_embeddings(keywords), dim=0))
        return torch.stack(words)

    def score_description_similarity(self, words):
        prediction = self.predict_descriptions(self.ent_embeddings.weight)
        vectors = torch.mean(self.word_embeddings(words), dim=1)

        # similarities = vectors.unsqueeze(0) @ prediction.transpose(-1, -2)
        # similarities = similarities.squeeze(0)
        # return similarities

        vec_norm = torch.linalg.norm(vectors, dim=1, keepdim=True)  # [batch_size, 1]
        pred_norm = torch.linalg.norm(prediction.T, dim=0, keepdim=True)  # [1, nentity]

        cos_sim = ((vectors @ prediction.T) / (vec_norm @ pred_norm))
        return cos_sim

    def loss(self, data, attr_loss_fn, alpha) -> Tensor:
        if False:  # train without relational data
            triples = torch.cat((data['batch_h'].unsqueeze(1), data['batch_r'].unsqueeze(1), data['batch_t'].unsqueeze(1)), -1)
            (scores_o, scores_s), factors = self.score_candidates(triples)
            l_fit = self.loss_fn(scores_o, triples[:, 2]) + self.loss_fn(scores_s, triples[:, 0])
            l_reg = self.regularizer.forward(factors)

        desc_e_emb = self.ent_embeddings(data['batch_h'])
        rel_emb = self.rel_embeddings(data['batch_r'])
        lhs = desc_e_emb[..., :self.rank], desc_e_emb[..., self.rank:]
        rel = rel_emb[..., :self.rank], rel_emb[..., self.rank:]

        pred = torch.cat(((lhs[0] * rel[0] - lhs[1] * rel[1]), (lhs[1] * rel[0] + lhs[0] * rel[1])), dim=1)
        scores = pred.unsqueeze(0) @ self.word_embeddings.weight.transpose(-1, -2)
        scores = scores.squeeze(0)
        # Compute ce loss to all word embeddings (as in RLKB)
        temp = F.log_softmax(scores, dim=-1)
        scores = list()
        for i in range(data['batch_t'].shape[0]):
            e = data['batch_t'][i]
            keywords = self.keywords[e.item()]
            if type(self.keywords[e.item()]) != bool:  # if entity has a description
                scores.append(torch.sum(temp[i][keywords]))

        return -sum(scores) / len(scores)
