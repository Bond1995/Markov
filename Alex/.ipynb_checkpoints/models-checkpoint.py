import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_embd = config.n_embd
        self.n_head = 1
        self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                             .view(1, 1, config.sequence_length, config.sequence_length))


    def forward(self, x):
        B, T, C = x.size()
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs);

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        
        return x


class Block(nn.Module):

    def __init__(self, config):
        super(Block, self).__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        att_x = self.attn(x)
        x = x + att_x
        contrib_norm = (att_x.norm(dim=2) / x.norm(dim=2)).mean().item()
        x = x + self.mlp(x)

        return x, contrib_norm


class GPTBase(nn.Module):

    def __init__(self, config):
        super(GPTBase, self).__init__()
        assert config.sequence_length is not None
        self.config = config
        self.b = nn.Parameter(torch.zeros(1))
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(1, config.n_embd, bias=False),
            wpe = nn.Embedding(config.sequence_length, config.n_embd),
            h = Block(config)
        ))

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, get_logits=False):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0)
        tok_emb = self.transformer.wte(idx.unsqueeze(-1).type(torch.float32)) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        x, contrib_norm = self.transformer.h(x)
        # logits = self.lm_head(x)
        logits = F.linear(x, self.transformer.wte.weight.t(), bias=self.b).squeeze(-1)
        return logits, contrib_norm
