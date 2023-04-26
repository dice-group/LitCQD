import torch
import torch.nn.functional as F

from torch import Tensor

from models.CQDComplExD import CQDComplExD


class CQDComplExAD(CQDComplExD):
    """
    ComplEx-N3 + Attribute Model + Descriptions model
    """

    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super(CQDComplExAD, self).__init__(*args, **kwargs)

    def loss(self, data, attr_loss_fn, alpha,beta = 1.0) -> Tensor:
        triples = torch.cat((data['batch_h'].unsqueeze(1), data['batch_r'].unsqueeze(1), data['batch_t'].unsqueeze(1)), -1)
        (scores_o, scores_s), factors = self.score_candidates(triples)
        l_fit = self.loss_fn(scores_o, triples[:, 2]) + self.loss_fn(scores_s, triples[:, 0])
        l_reg = self.regularizer.forward(factors)

        attr_scores, attr_factors = self.score_attr(data)
        attr_loss = attr_loss_fn(attr_scores)
        attr_reg = self.regularizer.forward(attr_factors)

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

        # return alpha * (l_fit + l_reg) + alpha * (attr_loss + attr_reg) + alpha * (desc_loss + desc_reg)
        return l_fit + l_reg + alpha * (attr_loss + attr_reg) + beta * (desc_loss + desc_reg)
        