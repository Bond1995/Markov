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

class AddBeta():
    """ Add-beta estimator. """

    def __init__(self, beta, shape, device):
        self.beta = beta
        self.counts = torch.zeros(shape, device=device)
        self.device = device

    def train(self, x):
        # Zero state counts
        y = (x[:,:-1] == 0)
        z = x[:,1:][y]
        self.counts[0,0] += z.numel() - z.sum()
        self.counts[0,1] += z.sum()

        # One state counts
        y = (x[:,:-1] == 1)
        z = x[:,1:][y]
        self.counts[1,0] += z.numel() - z.sum()
        self.counts[1,1] += z.sum()

    def estimate(self):
        return F.normalize(self.counts + self.beta, p=1.0, dim=1)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                                        .view(1, 1, config.sequence_length, config.sequence_length))

        self.memory = config.memory
        self.device = config.device
        self.wandb = config.wandb
        self.config = config
        self.iter = 0

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        if self.iter == 10000:
            print(self.c_attn.weight.cpu().detach().type(torch.float).numpy())
            fig_qkv = self.c_attn.weight.cpu().detach().type(torch.float)
            fig_qkv = (fig_qkv - fig_qkv.min()) / (fig_qkv.max()-fig_qkv.min())
            plt.imshow(fig_qkv.numpy(), cmap='gray', interpolation='nearest')
            if self.wandb:
                wandb.log({"att-qkv-"+str(self.iter): plt})
            
            print("W_0 * W_V:")
            prod = self.c_proj.weight[:,:] @ self.c_attn.weight[2*C:,:]
            prod = prod.cpu().detach().type(torch.float)
            print(prod.numpy())
            fig_prod = (prod - prod.min()) / (prod.max()-prod.min())
            plt.imshow(fig_prod.numpy(), cmap='gray', interpolation='nearest')
            if self.wandb:
                wandb.log({"prod-"+str(self.iter): plt})

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
            
            if self.iter == 10000:
                print("q,k:")
                print(q[0,0,:,:].cpu().detach().type(torch.float).numpy())
                print(k[0,0,:,:].cpu().detach().type(torch.float).numpy())
                mat = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                print("mat:")
                print(mat[0,0,:,:].cpu().detach().type(torch.float).numpy())
                print(mat[0,0,-1,:].cpu().detach().type(torch.float).numpy())
                attn_bias = torch.zeros(T, T, dtype=q.dtype, device=mat.device)
                temp_mask = torch.ones(T, T, dtype=torch.bool, device=mat.device).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                mat += attn_bias
                mat = F.softmax(mat, dim=-1)
                print(mat[0,0,-1,:].cpu().detach().type(torch.float).numpy())
                fin = mat @ v
                fig_att = mat[0,0,:100,:].cpu().detach()
                fig_att = (fig_att - fig_att.min()) / (fig_att.max()-fig_att.min())
                plt.imshow(fig_att.numpy(), cmap='gray', interpolation='nearest')
                if self.wandb:
                    wandb.log({"att-mat-"+str(self.iter): plt})
                fig_v = v[0,0,:100,:].cpu().detach().type(torch.float)
                fig_v = (fig_v - fig_v.min()) / (fig_v.max()-fig_v.min())
                plt.imshow(fig_v.numpy(), cmap='gray', interpolation='nearest')
                if self.wandb:
                    wandb.log({"att-value-"+str(self.iter): plt})
                fig_out = fin[0,0,:100,:].cpu().detach().type(torch.float)
                fig_out = (fig_out - fig_out.min()) / (fig_out.max()-fig_out.min())
                plt.imshow(fig_out.numpy(), cmap='gray', interpolation='nearest')
                if self.wandb:
                    wandb.log({"att-out-"+str(self.iter): plt})

                print("att_mat:")
                print(mat[0,0])
                print("att-value:")
                print(v[0,0])
                print("att_out:")
                print(fin[0,0])
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
        if self.iter == 10000:
            print("att_proj:")
            print(self.c_proj.weight)
            fig_att_proj = self.c_proj.weight[:,:].cpu().detach()
            fig_att_proj = (fig_att_proj - fig_att_proj.min()) / (fig_att_proj.max()-fig_att_proj.min())
            plt.imshow(fig_att_proj.numpy(), cmap='gray', interpolation='nearest')
            if self.wandb:
                wandb.log({"att-proj-"+str(self.iter): plt})

            # fig_y = y[:,:].cpu().detach()
            # fig_y = (fig_y - fig_y.min()) / (fig_y.max()-fig_y.min())
            # plt.imshow(fig_y.numpy(), cmap='gray', interpolation='nearest')
            # if self.wandb:
            #     wandb.log({"att-y-"+str(self.iter): plt})
        self.iter += 1

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.ReLU() #GELU
        self.config = config
        self.iter = 0
        self.wandb = config.wandb

    def forward(self, x):
        x = self.c_fc(x)
        if self.iter == 10000:
            print("y:")
            print(x[0].cpu().detach().type(torch.float).numpy())
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        if self.iter == 10000:
            print("w1_avg:")
            print(torch.mean(torch.abs(self.c_fc.weight)))
            print("c_fc:")
            print(self.c_fc.weight)
            print("c_proj:")
            print(self.c_proj.weight)
            fig_c_fc = self.c_fc.weight[:,:].cpu().detach()
            fig_c_fc = (fig_c_fc - fig_c_fc.min()) / (fig_c_fc.max()-fig_c_fc.min())
            plt.imshow(fig_c_fc.numpy(), cmap='gray', interpolation='nearest')
            if self.wandb:
                wandb.log({"c_fc-"+str(self.iter): plt})
            fig_c_proj = self.c_proj.weight.cpu().detach()
            fig_c_proj = (fig_c_proj - fig_c_proj.min()) / (fig_c_proj.max()-fig_c_proj.min())
            plt.imshow(fig_c_proj.numpy(), cmap='gray', interpolation='nearest')
            if self.wandb:
                wandb.log({"c_proj-"+str(self.iter): plt})
            corr = self.c_proj.weight[:,:] @ self.c_fc.weight[:,:]
            print("corr:")
            print(corr)
            fig_corr = corr.cpu().detach()
            fig_corr = (fig_corr - fig_corr.min()) / (fig_corr.max()-fig_corr.min())
            plt.imshow(fig_corr.numpy(), cmap='gray', interpolation='nearest')
            if self.wandb:
                wandb.log({"corr-"+str(self.iter): plt})
        self.iter += 1
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        #self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        #self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.iter = 0

    def forward(self, x):
        if self.iter == 10000:
            y = self.attn(x)
            z = x + y
            print("Approximation error:")
            print(torch.mean(torch.norm(y[0], dim=1) / torch.norm(z[0], dim=1)))
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
        self.iter = 0
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(1, config.n_embd, bias=False), # changed!
            wpe = nn.Embedding(config.sequence_length, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            #ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        if self.config.no_tying:
            self.lm_head = nn.Linear(config.n_embd, 1, bias=True) # changed! * 2
        else:
            if self.config.init == "ashok":
                self.b = nn.Parameter(torch.ones(1) * np.log(self.config.p / self.config.q))
            else:
                self.b = nn.Parameter(torch.zeros(1))
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        #self.transformer.wte.weight.data = self.lm_head.weight.data.view(self.lm_head.weight.shape[1], self.lm_head.weight.shape[0]) # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        if self.config.init == "ashok":
            for pn, p in self.named_parameters():
                if pn.endswith('wte.weight'):
                    torch.nn.init.zeros_(p)
                elif pn.endswith('c_fc.weight'):
                    torch.nn.init.zeros_(p)
                elif pn.endswith('c_proj.weight'):
                    torch.nn.init.zeros_(p)
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
        if self.iter == 10000:
            print(idx[0,:100])
            plt.imshow(torch.unsqueeze(idx[0,:100],0).cpu().detach().numpy(), cmap='gray', interpolation='nearest')
            if self.wandb:
                wandb.log({"idx-"+str(self.iter): plt})
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx.unsqueeze(-1).type(torch.float32)) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        if self.iter == 10000:
            print("e_avg:")
            print(torch.mean(torch.abs(self.transformer.wte.weight)))
            print("p_avg:")
            print(torch.mean(torch.abs(self.transformer.wpe.weight)))
            print("wte:")
            print(self.transformer.wte.weight)
            print("wpe:")
            print(self.transformer.wpe.weight)
            fig_wte = self.transformer.wte.weight[:,:].cpu().detach()
            fig_wte = (fig_wte - fig_wte.min()) / (fig_wte.max()-fig_wte.min())
            plt.imshow(fig_wte.numpy(), cmap='gray', interpolation='nearest')
            if self.wandb:
                wandb.log({"wte-"+str(self.iter): plt})
            fig_wpe = self.transformer.wpe.weight[:,:].cpu().detach().type(torch.float)
            fig_wpe = (fig_wpe - fig_wpe.min()) / (fig_wpe.max()-fig_wpe.min())
            plt.imshow(fig_wpe.numpy(), cmap='gray', interpolation='nearest')
            if self.wandb:
                wandb.log({"wpe-"+str(self.iter): plt})

            print("X_0 and X_1:")
            print(self.transformer.wte.weight.flatten() + self.transformer.wpe.weight[0,:].flatten())
            print(self.transformer.wpe.weight[0].flatten())

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        #x = self.transformer.ln_f(x) # (b, t, n_embd)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            if self.config.no_tying:
                logits = self.lm_head(x).squeeze(-1) # (b, t)
                if self.iter == 10000:
                    print("lm_head:")
                    print(self.lm_head.weight)
                    print("bias:")
                    print(self.lm_head.bias)
                    fig_lm_head = self.lm_head.weight[:,:].cpu().detach()
                    fig_lm_head = (fig_lm_head - fig_lm_head.min()) / (fig_lm_head.max()-fig_lm_head.min())
                    plt.imshow(fig_lm_head.numpy(), cmap='gray', interpolation='nearest')
                    if self.wandb:
                        wandb.log({"lm_head-"+str(self.iter): plt})
                    fig_bias = self.lm_head.bias.unsqueeze(0)[:,:].cpu().detach()
                    fig_bias = (fig_bias - fig_bias.min()) / (fig_bias.max()-fig_bias.min())
                    plt.imshow(fig_bias.numpy(), cmap='gray', interpolation='nearest')
                    if self.wandb:
                        wandb.log({"bias-"+str(self.iter): plt})
            else:
                logits = F.linear(x, self.transformer.wte.weight.t(), bias=self.b).squeeze(-1) # (b,t)
                if self.iter == 10000:
                    print("lm_head:")
                    print(self.transformer.wte.weight.t())
                    print("bias:")
                    print(self.b)
                    fig_lm_head = self.transformer.wte.weight.t()[:,:].cpu().detach()
                    fig_lm_head = (fig_lm_head - fig_lm_head.min()) / (fig_lm_head.max()-fig_lm_head.min())
                    plt.imshow(fig_lm_head.numpy(), cmap='gray', interpolation='nearest')
                    if self.wandb:
                        wandb.log({"lm_head-"+str(self.iter): plt})
                    fig_bias = self.b.unsqueeze(0)[:,:].cpu().detach()
                    fig_bias = (fig_bias - fig_bias.min()) / (fig_bias.max()-fig_bias.min())
                    plt.imshow(fig_bias.numpy(), cmap='gray', interpolation='nearest')
                    if self.wandb:
                        wandb.log({"bias-"+str(self.iter): plt})
            loss = F.binary_cross_entropy_with_logits(logits.view(-1), targets.float().view(-1))
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        logits = logits if get_logits else None

        # Compute Hessian matrix
        p_avg = torch.mean(self.transformer.wpe.weight, dim=0)
        hess = torch.eye(self.config.n_embd, device=p_avg.device) * 2 * (self.config.p + self.config.q - 1)
        for i in range(self.config.sequence_length):
            v = self.transformer.wpe.weight[i,:] - p_avg
            hess += v.unsqueeze(1) @ v.unsqueeze(0) / self.config.sequence_length
        eig = torch.linalg.eigvalsh(hess)[0]
        
        self.iter += 1
        return {'logits': logits, 'loss': loss, 'eig': eig}

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
