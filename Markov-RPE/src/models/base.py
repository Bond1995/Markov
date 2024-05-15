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
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # key and value embeddings
        self.embk = nn.Parameter(0.02 * torch.randn(config.sequence_length, config.n_embd))
        self.embv = nn.Parameter(0.02 * torch.randn(config.sequence_length, config.n_embd))
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.id = id
        self.iterations = config.iterations
        self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                                        .view(1, 1, config.sequence_length, config.sequence_length))
        self.memory = config.memory
        self.device = config.device
        self.wandb = config.wandb
        self.iter = 1

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        if self.iter == 2*self.iterations:
            
            print("id" + str(self.id) + " W_Q W_K W_V:")
            print(self.c_attn.weight.cpu().detach().type(torch.float).numpy())
            fig_qkv = self.c_attn.weight.cpu().detach().type(torch.float)
            fig_qkv = (fig_qkv - fig_qkv.min()) / (fig_qkv.max()-fig_qkv.min())
            plt.imshow(fig_qkv.numpy(), cmap='gray', interpolation='nearest')
            if self.wandb:
                wandb.log({"id" + str(self.id) + "-att-qkv-"+str(self.iter): plt})
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q1 = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q2 = q.permute(1, 0, 2).contiguous().view(T, B * self.n_head, C // self.n_head)
        
        relative_pos = torch.arange(T)[:, None] - torch.arange(T)[None, :]
        relative_pos = relative_pos.tril().int().to(self.device)
        rk = self.embk[relative_pos]
        rv = self.embv[relative_pos]

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        '''if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # Memory attention mask
            if self.memory >= 0:
                M1 = torch.ones(T, T, dtype=torch.bool).tril(diagonal=0)
                M2 = torch.ones(T, T, dtype=torch.bool).tril(diagonal=-self.memory-1)
                attn_mask = M1 * (~M2)
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask.to(self.device), dropout_p=self.dropout)
            else:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)'''
        # manual implementation of attention (no flash attention with relative positional embeddings)
        att1 = q1 @ k.transpose(-2, -1)
        att2 = q2 @ rk.transpose(1, 2)
        att2 = att2.transpose(0, 1).contiguous().view(B, self.n_head, T, T)
        att = (att1 + att2) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y1 = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y2 = att.permute(2, 0, 1, 3).contiguous().view(T, B * self.n_head, T)
        y2 = y2 @ rv
        y2 = y2.transpose(0, 1).contiguous().view(B, self.n_head, T, C // self.n_head)
        y = y1 + y2
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if self.iter == 2*self.iterations:
            print("id" + str(self.id) + "_att_proj:")
            print(self.c_proj.weight)
        
        self.iter += 1

        return y


class MLP(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
        self.id = id
        self.iterations = config.iterations
        self.iter = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        if self.iter == 2*self.iterations:
            print("id" + str(self.id) + "_c_fc:")
            print(self.c_fc.weight)
            print("id" + str(self.id) + "_c_proj:")
            print(self.c_proj.weight)
        
        self.iter += 1

        return x


class Block(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(id, config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(id, config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ln_2(self.mlp(x))
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
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(id, config) for id in range(config.n_layer)]),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        if self.config.init == "ashok":
            for pn, p in self.named_parameters():
                if pn.endswith('mlp.c_fc.weight'):
                    torch.nn.init.constant_(p, config.init_value)
                elif pn.endswith('mlp.c_proj.weight'):
                    torch.nn.init.constant_(p, -config.init_value)
                elif pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
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
        if self.iter == 2*self.iterations:
            print("Input sequence (first 100 samples):")
            print(idx[0,:100])
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.iter == 2*self.iterations:
            print("wte:")
            print(self.transformer.wte.weight)
            print("wpe:")
            print(self.transformer.wpe.weight)

        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) # (b, t, vocab_size)
            if self.iter == 2*self.iterations:
                print("lm_head:")
                print(self.lm_head.weight)
            #logits = F.normalize(F.relu(logits) + 1e-5, p=1.0, dim=-1)
            #logits = torch.log(logits)
            #loss = F.nll_loss(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
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
                elif pn.endswith('embk') or pn.endswith('embv'):
                    no_decay.add(fpn)

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
