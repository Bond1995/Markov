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
import os

import tiktoken
import torch
import wandb
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F





def normalize_data(data):
    try:
        # Ensure the data is a numpy array
        data = np.array(data)
        # Normalize between 0 and 1
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return normalized_data
    except Exception as e:
        return 0
    
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
        self.iter = 1
        self.eval_freq = config.eval_freq
        self.ckpt_path = None

    def get_qkv(self):
        q, k, v = self.c_attn.weight.T.split(self.n_embd, dim=1)
        return q, k, v


    def log_energy_and_weights(self, weight, wandb_name: str, iter: int) -> None:
        weight = weight.clone().detach()
        sv = torch.linalg.svdvals(weight)
        energy1 = sv[0]**2 / torch.sum(sv**2)
        energy2 = torch.sum(sv[:2]**2) / torch.sum(sv**2)
        energy3 = torch.sum(sv[:3]**2) / torch.sum(sv**2)
        energy4 = torch.sum(sv[:4]**2) / torch.sum(sv**2)
        energy5 = torch.sum(sv[:5]**2) / torch.sum(sv**2)
        if self.wandb:
            wandb.log({
                f"{wandb_name}-energy1": energy1.item(),
                f"{wandb_name}-energy2": energy2.item(),
                f"{wandb_name}-energy3": energy3.item(),
                f"{wandb_name}-energy4": energy4.item(),
                f"{wandb_name}-energy5": energy5.item(),
            })
        
        numpy_weight = weight.cpu().numpy()    
        self.save_weights(numpy_weight, wandb_name, iter)
    
    def visualize_weights(self, weight, wandb_name: str, iter: int) -> None:
        weight = weight.detach().clone()
        if self.wandb:
            wandb.log({f"{wandb_name}-iter{self.iter}-weight": wandb.Image(weight.cpu().numpy())})

    def save_weights(self, weight, wandb_name: str, iter: int, folder_name=False, save_folder=False) -> None:
        if self.wandb:
            run_dir = wandb.run.dir
            if save_folder and folder_name is not None:
                weight_folder_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/weights"
            else:
                weight_folder_path = f"{self.ckpt_path}/{wandb_name}/weights"
            os.makedirs(weight_folder_path, exist_ok=True)
            np.save(f"{weight_folder_path}/{wandb_name}-iter{self.iter}.npy", weight)

    def save_images(self, weight, wandb_name: str, iter: int, folder_name=False, save_folder=False) -> None:
        if self.wandb:
            run_dir = wandb.run.dir
            if save_folder and folder_name is not None:
                weight_images_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/images"
                try:
                    wandb.log({f"{folder_name}/{wandb_name}-iter{self.iter}-weight": plt.imshow(weight, cmap='viridis', interpolation='nearest')})
                except Exception as e:
                    print(f"Error: {e}")
            else:
                weight_images_path = f"{self.ckpt_path}/{wandb_name}/images"
            os.makedirs(weight_images_path, exist_ok=True)
            plt.figure()
            heatmap = plt.imshow(weight, cmap='viridis', interpolation='nearest')
            plt.colorbar(heatmap)
            plt.savefig(f"{weight_images_path}/{wandb_name}-iter{self.iter}.png")
            plt.close()
            
            if self.wandb:
                wandb.log({f"{wandb_name}/{wandb_name}-iter{self.iter}": wandb.Image(f"{weight_images_path}/{wandb_name}-iter{self.iter}.png")})
 
    
    def forward(self, x, get_att=True, folder_name=None, save_forward=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        if (self.iter == 1) or (self.iter % self.eval_freq == 0):
            q, k, v = self.get_qkv()
            self.log_energy_and_weights(q, f"q-id{self.id}", self.iter)
            self.log_energy_and_weights(k, f"k-id{self.id}", self.iter)
            self.log_energy_and_weights(v, f"v-id{self.id}", self.iter)
            
            proj_weight = self.c_proj.weight.T.detach().clone()
            self.log_energy_and_weights(proj_weight, f"proj-id{self.id}", self.iter)
            
            if self.c_proj.bias is not None:
                proj_bias = self.c_proj.bias.detach().clone()
                if self.wandb:
                    wandb.log({f"proj-id{self.id}-bias": proj_bias.cpu().numpy()})
                    wandb.save(f"proj-id{self.id}-bias.pt.npy", proj_bias.cpu().numpy(force=True))
            
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # Memory attention mask
            if self.memory >= 0:
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

        get_att = True
        if get_att:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(torch.tril(torch.ones(T, T, device=self.device)).view(1, 1, T, T) == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att_mean = att
            att_std = att.std(dim=0)
            
            att_filtered = att.clone().detach().cpu().numpy()
            
            if self.iter == 1 or self.iter % self.eval_freq == 0 or save_forward==True:
                self.save_images(att_filtered[0, 0, 0:64, 0:64], "att-filtered-id" + str(self.id), self.iter, folder_name, save_forward)
                self.save_weights(att_filtered, "att-id" + str(self.id), self.iter, folder_name, save_forward)

                if save_forward == True or self.iter % (self.eval_freq) == 0:
                    self.save_images(att_filtered[0, 0, :, :], "att-id" + str(self.id), self.iter, folder_name, save_forward)

            # np.save('att_mean_'+str(self.id)+'.pt', att_mean.numpy(force=True))
            # if self.wandb:
            #     wandb.save('att_mean_'+str(self.id)+'.pt.npy')
            # np.save('att_std_'+str(self.id)+'.pt', att_std.numpy(force=True))
            # if self.wandb:
            #     wandb.save('att_std_'+str(self.id)+'.pt.npy')
        else:
            att_mean = None
            att_std = None

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        if self.iter == self.iterations + np.floor(self.iterations/self.config.eval_freq):
            print("id" + str(self.id) + "_att_proj:")
            print(self.c_proj.weight)
        

        return y, att_mean, att_std


class MLP(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
        self.id = id
        self.wandb = config.wandb
        self.iterations = config.iterations
        self.config = config
        self.iter = 1
        self.eval_freq = config.eval_freq
        self.ckpt_path = None
        
        

    def save_weights(self, weight, wandb_name: str, iter: int, folder_name, save_forward) -> None:
        if self.wandb:
            run_dir = wandb.run.dir
            if save_forward and folder_name is not None:
                weight_folder_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/weights"
            else:
                weight_folder_path = f"{self.ckpt_path}/{wandb_name}/weights"
            
            os.makedirs(weight_folder_path, exist_ok=True)
            np.save(f"{weight_folder_path}/{wandb_name}-iter{self.iter}.npy", weight)


    def forward(self, x, folder_name=None, save_forward=False):
        weights_layer1 = self.c_fc.weight.clone().detach().cpu().numpy()
        weights_layer2 = self.c_proj.weight.clone().detach().cpu().numpy()
        if self.iter == 1 or self.iter % self.eval_freq == 0 or save_forward == True:
            self.save_weights(weights_layer1, "mlp-c_fc-id" +str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(weights_layer2, "mlp-c_proj-id" + str(self.id), self.iter, folder_name=folder_name, save_forward=save_forward)
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        
        # self.iter += 1

        return x


class Block(nn.Module):

    def __init__(self, id, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(id, config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(id, config)

    def forward(self, x, get_att=False, folder_name=None, save_forward=False):
        z, att_mean, att_std = self.attn(self.ln_1(x), get_att=get_att, folder_name=folder_name, save_forward=save_forward)
        x = x + z
        x = x + self.mlp(self.ln_2(x), folder_name=folder_name, save_forward=save_forward)
        return x, att_mean, att_std
    

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
        self.eval_freq = config.eval_freq
        self.ckpt_path = None
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.sequence_length, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(id, config) for id in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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
    
    def update_iter(self):
        self.iter += 1
        for block in self.transformer.h:
            block.attn.iter = self.iter
            block.mlp.iter = self.iter
    
    def update_ckpt_path(self, ckpt_path):
        for block in self.transformer.h:
            block.attn.ckpt_path = ckpt_path
            block.mlp.ckpt_path = ckpt_path

    def save_weights(self, weight, wandb_name: str, iter: int, folder_name=None, save_forward=False) -> None:
        if self.wandb:
            run_dir = wandb.run.dir
            if save_forward and folder_name is not None:
                weight_folder_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/weights"
            else:
                weight_folder_path = f"{self.ckpt_path}/{wandb_name}/weights"
            os.makedirs(weight_folder_path, exist_ok=True)
            np.save(f"{weight_folder_path}/{wandb_name}-iter{self.iter}.npy", weight)

    def save_images(self, weight, wandb_name: str, iter: int, folder_name=None, save_forward=False) -> None:
        if self.wandb:
            run_dir = wandb.run.dir
            if save_forward and folder_name is not None:
                weight_images_path = f"{self.ckpt_path}/{folder_name}/{wandb_name}/images"
                try:
                    wandb.log({f"{folder_name}/{wandb_name}-iter{self.iter}-weight": plt.imshow(weight, cmap='viridis', interpolation='nearest')})
                except Exception as e:
                    print(f"Error: {e}")
            else:
                weight_images_path = f"{self.ckpt_path}/{wandb_name}/images"
            os.makedirs(weight_images_path, exist_ok=True)
            plt.figure()
            heatmap=plt.imshow(weight, cmap='viridis', interpolation='nearest')
            plt.colorbar(heatmap)
            plt.savefig(f"{weight_images_path}/{wandb_name}-iter{self.iter}.png")
            plt.close()
            
            if self.wandb:
                wandb.log({f"{wandb_name}/{wandb_name}-iter{self.iter}": wandb.Image(f"{weight_images_path}/{wandb_name}-iter{self.iter}.png")})

    def plot_images_on_wandb(self, weight, wandb_name: str, iter: int) -> None:
        if self.wandb:
            wandb.log({f"{wandb_name}-iter{self.iter}-weight": plt.imshow(weight, cmap='viridis', interpolation='nearest')}) 

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

    def forward(self, idx, targets=None, get_logits=False, get_att=False, folder_name=None, save_forward = False):
        run_dir = wandb.run.dir
        device = idx.device
        b, t = idx.size()
        if self.iter == self.iterations + np.floor(self.iterations/self.config.eval_freq):
            print("Input sequence (first 100 samples):")
            print(idx[0,:100])
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
        # save tok_emb as a heatmap
        if self.iter == 1 or self.iter % self.eval_freq == 0 or save_forward == True:
            copied_token_embedding_weights = self.transformer.wte.weight.clone().detach().cpu().numpy()
            # self.plot_images_on_wandb(copied_token_embedding_weights, "token_embeddings/tok_emb", self.iter)
            copied_token_embedding_weights = normalize_data(copied_token_embedding_weights)
            self.save_images(copied_token_embedding_weights, "token_embeddings", self.iter, folder_name=folder_name, save_forward=save_forward)
            self.save_weights(copied_token_embedding_weights, "token_embeddings", self.iter, folder_name=folder_name, save_forward=save_forward)
                        
            copied_positional_embedding_weights = self.transformer.wpe.weight.clone().detach().cpu().numpy()
            wandb.log({f"positional_embeddings/pos_emb-iter{self.iter}": plt.imshow(copied_positional_embedding_weights,
                                                                                    cmap='viridis', interpolation='nearest')})
            # self.plot_images_on_wandb(copied_positional_embedding_weights, "positional_embeddings/pos_emb", self.iter)
            copied_positional_embedding_weights = normalize_data(copied_positional_embedding_weights)
            self.save_images(copied_positional_embedding_weights, "positional_embeddings", self.iter)
            self.save_weights(copied_positional_embedding_weights, "positional_embeddings", self.iter)
        
        if self.iter == self.iterations + np.floor(self.iterations/self.config.eval_freq):
            print("wte:")
            print(self.transformer.wte.weight)
            print("wpe:")
            print(self.transformer.wpe.weight)

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x, att_mean, att_std = block(x, get_att=get_att, folder_name=folder_name, save_forward=save_forward)
        x = self.transformer.ln_f(x) # (b, t, n_embd)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) # (b, t, vocab_size)
            if self.iter == self.iterations + np.floor(self.iterations/self.config.eval_freq):
                print("lm_head:")
                print(self.lm_head.weight)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        logits = logits if get_logits else None
        att_mean = att_mean if get_att else None
        att_std = att_std if get_att else None

        # self.iter += 1

        return {'logits': logits, 'loss': loss, 'att_mean': att_mean, 'att_std': att_std}

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
