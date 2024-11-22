import math

import torch
import wandb
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
    

class MarkovFixed(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.wandb = config.wandb
        self.iter = 0
        
        self.generator = torch.Generator()
        self.generator.seed()

        self.e = nn.Parameter(0.01*torch.randn(1, generator=self.generator) * torch.ones(1))
        self.w = nn.Parameter(0.01*torch.randn(1, generator=self.generator) * torch.ones(1))
        self.b = nn.Parameter(config.b * torch.ones(1))

        # report number of parameters
        print("number of parameters: %d" % (self.get_num_params()))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets, get_logits=False):
        _, t = idx.size()
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"

        logits = torch.square(self.e) * (1.0 + 2.0 * self.w * torch.abs(self.w)) * idx + self.b - 0.5 * torch.square(self.e)
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), targets.float().view(-1))
        logits = logits if get_logits else None

        if self.wandb:
            wandb.log({
                    "e_value": self.e.item(),
                    "w_value": self.w.item(),
                    "b_value": self.b.item(),
                })

        self.iter += 1

        return {'logits': logits, 'loss': loss}
