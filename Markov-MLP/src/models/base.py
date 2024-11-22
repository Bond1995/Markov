"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import tiktoken
import torch
import wandb
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.ReLU()
        self.config = config
        self.wandb = config.wandb
        self.iterations = config.iterations
        self.iter = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        if (self.iter == 1) or (self.iter % 100 == 0):
            print("c_fc:")
            print(self.c_fc.weight)
            print("c_proj:")
            print(self.c_proj.weight)

            if self.wandb:
                wandb.log({"c_fc-"+str(self.iter): wandb.Image(self.c_fc.weight.numpy(force=True))})
            if self.wandb:
                wandb.log({"c_proj-"+str(self.iter): wandb.Image(self.c_proj.weight.numpy(force=True))})

            np.save('c_fc-'+str(self.iter)+'.pt', self.c_fc.weight.numpy(force=True))
            if self.wandb:
                wandb.save('c_fc-'+str(self.iter)+'.pt.npy')
            
            sv = torch.linalg.svdvals(self.c_fc.weight.detach().cpu())
            energy1 = sv[0]**2 / torch.sum(sv**2)
            energy2 = torch.sum(sv[:2]**2) / torch.sum(sv**2)
            energy3 = torch.sum(sv[:3]**2) / torch.sum(sv**2)
            energy4 = torch.sum(sv[:4]**2) / torch.sum(sv**2)
            energy5 = torch.sum(sv[:5]**2) / torch.sum(sv**2)
            if self.wandb:
                wandb.log({
                    "train/c_fc_energy1": energy1.item(),
                    "train/c_fc_energy2": energy2.item(),
                    "train/c_fc_energy3": energy3.item(),
                    "train/c_fc_energy4": energy4.item(),
                    "train/c_fc_energy5": energy5.item(),
                })

        self.iter += 1

        return x
    

class GPTBase(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.wandb = config.wandb
        self.iterations = config.iterations
        self.iter = 1
        
        self.wte = nn.Linear(1, config.n_embd, bias=False) # changed!
        self.wpe = nn.Embedding(config.sequence_length, config.n_embd)
        self.mlp = MLP(config)

        if self.config.no_tying:
            self.lm_head = nn.Linear(config.n_embd, 1, bias=True) # changed! * 2
        else:
            self.b = nn.Parameter(torch.zeros(1))
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        #self.transformer.wte.weight.data = self.lm_head.weight.data.view(self.lm_head.weight.shape[1], self.lm_head.weight.shape[0]) # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, get_logits=False):
        device = idx.device
        b, t = idx.size()
        if (self.iter == 1) or (self.iter % 100 == 0):
            print(idx[0,:100])
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        
        tok_emb = self.wte(idx.unsqueeze(-1).type(torch.float32)) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        if (self.iter == 1) or (self.iter % 100 == 0):
            print("wte:")
            print(self.wte.weight)
            print("wpe:")
            print(self.wpe.weight)

            if self.wandb:
                wandb.log({"wte-"+str(self.iter): wandb.Image(self.wte.weight.numpy(force=True))})
            if self.wandb:
                wandb.log({"wpe-"+str(self.iter): wandb.Image(self.wpe.weight[:100].numpy(force=True))})
            
            np.save('wpe-'+str(self.iter)+'.pt', self.wpe.weight.numpy(force=True))
            if self.wandb:
                wandb.save('wpe-'+str(self.iter)+'.pt.npy')

        x = tok_emb + pos_emb
        x = self.mlp(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            if self.config.no_tying:
                logits = self.lm_head(x).squeeze(-1) # (b, t)
                if (self.iter == 1) or (self.iter % 100 == 0):
                    print("lm_head:")
                    print(self.lm_head.weight)
                    print("bias:")
                    print(self.lm_head.bias)

                    if self.wandb:
                        wandb.log({"lm_head-"+str(self.iter): wandb.Image(self.lm_head.weight.numpy(force=True))})
            else:
                logits = F.linear(x, self.wte.weight.t(), bias=self.b).squeeze(-1) # (b,t)
                if (self.iter == 1) or (self.iter % 100 == 0):
                    print("lm_head:")
                    print(self.wte.weight.t())
                    print("bias:")
                    print(self.b)
                    
                    if self.wandb:
                        wandb.log({"lm_head-"+str(self.iter): wandb.Image(self.wte.weight.t().numpy(force=True))})
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), targets.float().view(-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        logits = logits if get_logits else None
        self.iter += 1

        return {'logits': logits, 'loss': loss}

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        if not self.config.no_tying:
            no_decay.add('b')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]
