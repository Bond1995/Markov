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


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.id = id
        self.iterations = config.iterations
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                                        .view(1, 1, config.sequence_length, config.sequence_length))

        self.memory = config.memory
        self.device = config.device
        self.config = config
        self.wandb = config.wandb
        self.iter = 1

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        if (self.iter == 1) or (self.iter % 100 == 0):
            WV = self.c_attn.weight[2*self.config.n_embd:].detach().cpu()
            sv = torch.linalg.svdvals(WV)
            energy1 = sv[0]**2 / torch.sum(sv**2)
            energy2 = torch.sum(sv[:2]**2) / torch.sum(sv**2)
            energy3 = torch.sum(sv[:3]**2) / torch.sum(sv**2)
            energy4 = torch.sum(sv[:4]**2) / torch.sum(sv**2)
            energy5 = torch.sum(sv[:5]**2) / torch.sum(sv**2)
            if self.wandb:
                wandb.log({
                    "train/att-v-energy1": energy1.item(),
                    "train/att-v-energy2": energy2.item(),
                    "train/att-v-energy3": energy3.item(),
                    "train/att-v-energy4": energy4.item(),
                    "train/att-v-energy5": energy5.item(),
                })
            
            np.save('att-v-'+str(self.iter)+'.pt', WV.numpy(force=True))
            if self.wandb:
                wandb.save('att-v-'+str(self.iter)+'.pt.npy')

            

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs);

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # Memory attention mask
            if self.memory >= 0:
                print("Using masking!")
                M1 = torch.ones(T, T, dtype=torch.bool).tril(diagonal=0)
                M2 = torch.ones(T, T, dtype=torch.bool).tril(diagonal=-self.memory-1)
                attn_mask = M1 * (~M2)
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(self.device), dropout_p=self.dropout)
            else:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
            
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        self.iter += 1

        return y


class MLP(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.ReLU()
        self.config = config
        self.wandb = config.wandb
        self.id = id
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


class Block(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        #self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(id, config)
        #self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(id, config)
        self.iterations = config.iterations
        self.wandb = config.wandb
        self.iter = 1

    def forward(self, x):
        if (self.iter == 1) or (self.iter % 100 == 0):
            y = self.attn(x)
            z = x + y
            err = (y.norm(dim=2) / z.norm(dim=2)).mean()
            print("Approximation error:")
            print(err)

            if self.wandb:
                wandb.log({"val/approx-err": err.item()})

        x = x + self.attn(x)
        x = x + self.mlp(x)

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
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.sequence_length, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(id, config) for id in range(config.n_layer)]),
            #ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        if not self.config.no_tying:
            self.transformer.wte.weight = self.lm_head.weight
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        #self.transformer.wte.weight.data = self.lm_head.weight.data.view(self.lm_head.weight.shape[1], self.lm_head.weight.shape[0]) # https://paperswithcode.com/method/weight-tying

        self.register_buffer("alpha", 2 * torch.bernoulli(0.5*torch.ones(self.config.n_embd,1)) - 1)
        self.register_buffer("w", torch.abs(0.1*torch.randn(1)) * (2 * torch.bernoulli(0.5*torch.ones(4*self.config.n_embd,1)) - 1))
        # init all weights
        self.apply(self._init_weights)
        if self.config.init == "good":
            for pn, p in self.named_parameters():
                if pn.endswith('wte.weight'):
                    torch.nn.init.constant_(p, -1.0)
                elif pn.endswith('wpe.weight'):
                    torch.nn.init.constant_(p, 0.5)
                elif pn.endswith('mlp.c_fc.weight'):
                    torch.nn.init.constant_(p, 2)
                elif pn.endswith('mlp.c_proj.weight'):
                    torch.nn.init.constant_(p, -2)
        elif self.config.init == "lowrank":
            e = torch.abs(0.1*torch.randn(1))
            v = torch.randn(self.config.n_embd, 1) * self.config.v_std
            att_init = 0.02*torch.randn(3*self.config.n_embd,self.config.n_embd)
            att_init[2*self.config.n_embd:,:].copy_(self.alpha @ v.T)

            for pn, p in self.named_parameters():
                if pn.endswith('wte.weight'):
                    with torch.no_grad():
                        p.copy_(e.item()*self.alpha)
                if pn.endswith('wpe.weight'):
                    with torch.no_grad():
                        p.copy_(torch.ones(p.shape[0],1) @ (-0.5*e.item()*self.alpha.T))
                if pn.endswith('attn.c_attn.weight'):
                    with torch.no_grad():
                        p.copy_(att_init)
                if pn.endswith('mlp.c_fc.weight'):
                    with torch.no_grad():
                        p.copy_(self.w @ self.alpha.T)
                if pn.endswith('mlp.c_proj.weight'):
                    with torch.no_grad():
                        p.copy_(self.alpha @ self.w.T)
        else:
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
            n_params -= self.transformer.wpe.weight.numel()
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
        
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        if (self.iter == 1) or (self.iter % 100 == 0):
            print("wte:")
            print(self.transformer.wte.weight)
            print("wpe:")
            print(self.transformer.wpe.weight)

            if self.wandb:
                wandb.log({"wte-"+str(self.iter): wandb.Image(self.transformer.wte.weight.numpy(force=True))})
            if self.wandb:
                wandb.log({"wpe-"+str(self.iter): wandb.Image(self.transformer.wpe.weight[:100].numpy(force=True))})
            
            np.save('wpe-'+str(self.iter)+'.pt', self.transformer.wpe.weight.numpy(force=True))
            if self.wandb:
                wandb.save('wpe-'+str(self.iter)+'.pt.npy')

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        #x = self.transformer.ln_f(x) # (b, t, n_embd)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) # (b, t, vocab_size)
            if (self.iter == 1) or (self.iter % 100 == 0):
                print("lm_head:")
                print(self.lm_head.weight)
                print("bias:")
                print(self.lm_head.bias)

                if self.wandb:
                    wandb.log({"lm_head-"+str(self.iter): wandb.Image(self.lm_head.weight.numpy(force=True))})
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        logits = logits if get_logits else None
        self.iter += 1

        return {'logits': logits, 'loss': loss}

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:sequence_length])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:sequence_length,:sequence_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO
        pass

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
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
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
            decay.remove('lm_head.weight')

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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = torch.tensor(self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})).view(1,-1).to(self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)
