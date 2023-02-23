import statistics
import torch
import torch.nn as nn
import math

from models import CQDComplExA


class CQDComplExAWeighted(CQDComplExA):

    def __init__(self,
                 attr_values: dict = {},
                 *args,
                 **kwargs,
                 ):
        super(CQDComplExAWeighted, self).__init__(*args, **kwargs)

        if self.use_attributes:
            mads = list()
            for _, values in dict(sorted(attr_values.items())).items():
                try:
                    mads.append(sum([abs(statistics.mean(values)-v) for v in values])/len(values))
                except:
                    mads.append(1.0)
                if mads[-1] == 0.0:
                    mads[-1] = 1.0
                mads[-1] = 1/math.sqrt(mads[-1])
            self.attr_mads = nn.Embedding.from_pretrained(torch.FloatTensor(mads).unsqueeze(1))
            self.attr_mads.weight.requires_grad = False

    def score_attr(self, data):
        """
        Weight attributes with a low MAD more as these have a lower MAE when just predicting the mean value.
        """
        if "batch_e" not in data:
            return torch.FloatTensor().to(device=data['batch_h'].device), []
        e = self.ent_embeddings(data['batch_e'])
        multiplier = self.attr_mads(data['batch_a']).view(data['batch_a'].shape[0])
        v = data['batch_v'] * multiplier
        preds = self.predict_attribute_values(e, data['batch_a']) * multiplier

        factors = []
        for emb in (e, self.attr_embeddings(data['batch_a'])):
            emb_re, emb_im = emb[..., :self.rank], emb[..., self.rank:]
            factors.append(torch.sqrt(emb_re ** 2 + emb_im ** 2))

        return torch.stack((preds, v), dim=-1), factors
