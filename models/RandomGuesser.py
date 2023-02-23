from models.CQDBaseModel import CQDBaseModel

import torch
import torch.nn as nn
from torch import Tensor

from typing import Dict, Tuple


class RandomGuesser(nn.Module):
    """
    Unused. Just for Testing
    """

    def __init__(self, nentity
                 ):
        self.nentity = nentity
        super(RandomGuesser, self).__init__()

    #---------- Overridden methods ----------#

    def predict_attribute_values(self, e_emb, attributes):
        return torch.ones_like(attributes)*0.5

    def forward(self, batch_queries_dict: Dict[Tuple, Tensor]):
        all_scores = []

        for query_structure, queries in batch_queries_dict.items():
            if query_structure[-1] == ('ap', 'a'):
                pred = self.predict_attribute_values(queries[:, 0].long(), queries[:, 2].long())
                all_scores.extend([s for s in pred])
            else:
                all_scores.extend([torch.distributions.uniform.Uniform(0, 1).sample([self.nentity]) for i in range(queries.shape[0])])
        return all_scores
