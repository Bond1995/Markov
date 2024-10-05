import torch

import distributed


def parse_args(base_parser, args, namespace):
    parser = base_parser
    # General training params
    parser.add_argument('--batch_size', default=16, type=int) #50
    parser.add_argument('--acc_steps', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--iterations', default=10000, type=int)
    parser.add_argument('--lr', default=2e-3, type=float) #2e-3
    parser.add_argument('--warmup_percent', default=0.02, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.95, type=float)
    parser.add_argument('--scheduler', default='cos', choices=['linear', 'cos', 'none'])
    parser.add_argument('--opt', default='sgd', choices=['adamw', 'sgd'])
    parser.add_argument('--eval_freq', default=10, type=int) # in iterations
    parser.add_argument('--results_base_folder', default="./exps", type=str) 
    parser.add_argument('--grad_clip', default=1.0, type=float) # default value is 1.0 in NanoGPT
    # Dataset params
    parser.add_argument('--dataset', default='markov', choices=['wikitext', "shakespeare-char", 'arxiv', "arxiv2000", "arxiv+wiki", 'openwebtext2', 'markov'])
    parser.add_argument('--vocab_size', default=2, type=int)
    parser.add_argument('--data_in_ram', action='store_true') # force the data to RAM, mostly useless except for openwebtext2 
    # Model params
    parser.add_argument('--model', default='base', choices=['base', 'sparse-heads-q'])
    parser.add_argument('--use_pretrained', default="none", type=str) # 'none', 'gpt-2' or a path to the pretrained model
    parser.add_argument('--dropout', default=0, type=float) #0.2
    parser.add_argument('--n_head', default=1, type=int)
    parser.add_argument('--n_layer', default=1, type=int) # depths in att + ff blocks
    parser.add_argument('--n_embd', default=8, type=int) # embedding size / hidden size ... 
    parser.add_argument('--sequence_length', default=512, type=int)
    parser.add_argument('--dtype', default=torch.float32, type=torch.dtype) #changed!
    parser.add_argument('--bias', default=True, type=bool)
    parser.add_argument('--no_compile', default = True, action='store_true') # if true then model is not compiled 
    # logging params (WandB)
    parser.add_argument('--wandb', default=True, action='store_true') # whether to use wandb or not
    parser.add_argument('--wandb_project', default="bias-test", type=str)
    parser.add_argument('--wandb_run_prefix', default="none", type=str) # is added before the autogenerated experiment name
    parser.add_argument('--eval_seq_prefix', default="0", type=str) # prefix used to generate sequences
    # Distributed args
    parser.add_argument('--distributed_backend', default=None, type=str, required=False,
                        choices=distributed.registered_backends())  # distributed backend type
    # Markov args
    parser.add_argument('--p', default=0.5, type=float)
    parser.add_argument('--q', default=0.5, type=float)
    parser.add_argument('--order', default=1, type=int)
    parser.add_argument('--chain', default='random', choices=['switch', 'random'])
    # Memory args
    parser.add_argument('--memory', default=-1, type=int) # if negative, standard causal attention is applied
    # Starting distribution
    parser.add_argument('--initial', default='uniform', choices=['steady', 'uniform'])
    # Initialization args
    parser.add_argument('--init', default='base', choices=['base', 'ashok', 'lowrank'])
    parser.add_argument('--init_value', default=1.0, type=float)
    
    return parser.parse_args(args, namespace)
