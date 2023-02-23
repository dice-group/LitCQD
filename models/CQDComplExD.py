import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from models.CQDComplExA import CQDComplExA


class Gate(nn.Module):

    def __init__(self,
                 input_size,
                 output_size):

        super(Gate, self).__init__()
        self.output_size = output_size

        self.weight = nn.Linear(input_size, input_size)
        self.output = nn.Linear(input_size, output_size)

    def forward(self, x_ent):
        x_ent = torch.sigmoid(self.weight(x_ent)) * x_ent
        #x_ent = torch.softmax(self.weight(x_ent), dim=-1) * x_ent
        output = self.output(x_ent)

        return output


class CQDComplExD(CQDComplExA):
    """
    ComplEx-N3 + Descriptions model
    """

    def __init__(self,
                 word_emb_dim: int = 300,
                 desc_emb: str = '1-layer',
                 *args,
                 **kwargs,
                 ):
        super(CQDComplExD, self).__init__(*args, **kwargs)

        self.desc_emb = desc_emb
        if self.desc_emb == '1-layer':
            self.description_net = nn.Linear(2*self.rank, word_emb_dim, bias=True)
        elif self.desc_emb == '2-layer':
            self.description_net = torch.nn.Sequential(
                torch.nn.Linear(2*self.rank, self.rank),
                torch.nn.Linear(self.rank, word_emb_dim),
            )
        else:
            self.description_net = Gate(2*self.rank, word_emb_dim)
        self.dropout = nn.Dropout(0.0, inplace=False)

    def predict_descriptions(self, e_emb):
        return self.description_net(e_emb)

    def score_description_similarity(self, vectors):
        prediction = self.predict_descriptions(self.ent_embeddings.weight)

        vec_norm = torch.linalg.norm(vectors, dim=1, keepdim=True)  # [batch_size, 1]
        pred_norm = torch.linalg.norm(prediction.T, dim=0, keepdim=True)  # [1, nentity]
        cos_sim = ((vectors @ prediction.T) / (vec_norm @ pred_norm))
        return cos_sim

    def loss(self, data, attr_loss_fn, alpha) -> Tensor:
        assert self.use_attributes == False
        triples = torch.cat((data['batch_h'].unsqueeze(1), data['batch_r'].unsqueeze(1), data['batch_t'].unsqueeze(1)), -1)
        (scores_o, scores_s), factors = self.score_candidates(triples)
        l_fit = self.loss_fn(scores_o, triples[:, 2]) + self.loss_fn(scores_s, triples[:, 0])
        l_reg = self.regularizer.forward(factors)

        desc_e_emb = self.ent_embeddings(data['batch_desc_e'])
        pred = self.predict_descriptions(desc_e_emb)
        #desc_loss = torch.mean(F.relu(0.95-F.cosine_similarity(self.dropout(pred), data['batch_desc_d'])))
        desc_loss = torch.mean(1-F.cosine_similarity(self.dropout(pred), data['batch_desc_d']))

        if self.desc_emb == '1-layer':
            desc_factors = [self.description_net.weight, self.description_net.bias]
        elif self.desc_emb == '2-layer':
            desc_factors = [self.description_net[0].weight, self.description_net[0].bias, self.description_net[1].weight, self.description_net[1].bias]
        else:
            desc_factors = [self.description_net.weight.weight, self.description_net.weight.bias, self.description_net.output.bias, self.description_net.output.weight]
        desc_factors.append(torch.sqrt(desc_e_emb[..., :self.rank] ** 2 + desc_e_emb[..., self.rank:] ** 2))
        desc_reg = self.regularizer.forward(desc_factors)

        return (l_fit + l_reg) + (desc_loss + desc_reg)
