import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack


def optimal_est(P, order, x, y):
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(P.device)
    opt_logits = torch.zeros(x.size(0), x.size(1), P.size(1), device=P.device)
    if order > 1:
        opt_logits[:,:order-1,:] = 0.5*torch.ones(x.size(0), order-1, P.size(1), device=P.device)
    for i in range(order-1, x.size(1)):
        idx = x[:,i-order+1:i+1].float() @ powers
        opt_logits[:,i,:] = P[idx.to(int)]
    opt_logits = torch.log(opt_logits)
    opt_loss = F.nll_loss(opt_logits.view(-1, opt_logits.size(-1)), y.view(-1), ignore_index=-1)

    return opt_loss


def get_batch(P, order, seq_length, batch_size, generator, extra_args, device='cpu'):
    data = torch.zeros(batch_size, seq_length+1, device=device)
    if extra_args.initial == 'steady':
        if P.size(0) == 2:
            alpha = P[1,0] / (P[0,1] + P[1,0])
        else:
            alpha = 0.5
    elif extra_args.initial == 'uniform':
        alpha = 0.5
    else:
        alpha = 0.5
    # Generate first k bits
    for k in range(order):
        data[:,k] = torch.bernoulli(alpha*torch.ones((batch_size,), device=device), generator=generator)
    for i in range(order, seq_length):
        data[:,i] = get_next_symbols(P, order, data[:,i-order:i])
    x = data[:,:seq_length].to(int)
    y = data[:,1:].to(int)
    
    return x, y

def get_next_symbols(P, order, data):
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(data.device)
    idx = data @ powers
    M = P[idx.to(int)]
    s = torch.multinomial(M,1).flatten()

    return s


@torch.no_grad()
def eval(model, P, order, sequence_length, batch_size, generator, extra_args, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list, opt_loss_list = [], [], []

    for _ in range(max_num_batches): 
        x, y = get_batch(P, order, sequence_length, batch_size, generator, extra_args, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())
        opt_loss = optimal_est(P, order, x, y)
        opt_loss_list.append(opt_loss)

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss
    opt_loss = torch.stack(opt_loss_list).mean().item()

    return val_acc, val_loss, val_perplexity, opt_loss


@torch.no_grad()
def eval_att(model, P, order, sequence_length, batch_size, generator, extra_args, device='cpu', ctx=nullcontext()):
    assert model.training == False
    
    x, y = get_batch(P, order, sequence_length, batch_size, generator, extra_args, device=device)
    with ctx:
        outputs = model(x, targets=y, get_logits=True, get_att=True)
    att_mean = outputs['att_mean']
    att_std = outputs['att_std']

    return att_mean, att_std


@torch.no_grad()
def eval_probs(model, P, order, sequence_length, generator, extra_args, device='cpu', ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    x, y = get_batch(P, order, sequence_length, 1, generator, extra_args, device=device)
    with ctx:
        outputs = model (x, targets=y, get_logits=True)
    val_loss = outputs['loss']
    loss_list_val.append(val_loss)
    acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    probs = F.softmax(outputs['logits'], dim=-1)

    xb = x[0].float()
    probsb = probs[0, order-1:]
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(device)
    idx = torch.Tensor([xb[i:i+order] @ powers for i in range(sequence_length - order + 1)])
    prob_vec = []
    for i in range(2**order):
        vec = probsb[idx == i][:,1] # estimated p
        prob_vec.append(vec)

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity, prob_vec

@torch.no_grad()
def eval_sparse(model, P, sequence_length, batch_size, device='cpu', max_num_batches=24, ctx=nullcontext(), alpha_th=None, drop_k=None):
    assert model.training == False

    ce_loss_list_val, l1_loss_list_val, acc_list, sparcity_per_layer = [], [], [], []

    for _ in range(max_num_batches): 
        x, y = get_batch(P, sequence_length, batch_size, device=device)
        with ctx:
            outputs = model(x, targets=y, alpha_th=alpha_th, drop_k=drop_k, get_logits=True, get_alphas=True)
        ce_loss_list_val.append(outputs['ce_loss'])
        l1_loss_list_val.append(outputs['l1_loss'])
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())
        sparcity_per_layer.append([(alphas < 1e-8).sum().float().cpu().item() / alphas.numel() for alphas in outputs['alphas']])

    val_acc = torch.stack(acc_list).mean().item()
    val_ce_loss = np.mean(ce_loss_list_val)
    val_l1_loss = np.mean(l1_loss_list_val)
    val_perplexity = 2.71828 ** val_ce_loss
    sparcity_per_layer = np.mean(np.array(sparcity_per_layer), axis=0)

    return val_acc, val_ce_loss, val_l1_loss, val_perplexity, sparcity_per_layer


@torch.no_grad()
def eval_sweep_dropk(model, P, sequence_length, batch_size, n_heads, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    x_axis, y_axis_pp, y_axis_acc, y_axis_loss = torch.linspace(0.0,0.95,15), [], [], []
    loss_list_val, acc_list = [], []

    for frac in x_axis:
        drop_k = int(sequence_length * frac * n_heads)
        for _ in range(max_num_batches): 
            x, y = get_batch(P, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(x, targets=y, alpha_th=None, drop_k=drop_k, get_logits=True)
            loss_list_val.append(outputs['ce_loss'])
            acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


@torch.no_grad()
def eval_sweep_alphath(model, P, sequence_length, batch_size, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    alpha_ths, y_axis_pp, y_axis_acc, y_axis_loss = [0, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1], [], [], []
    loss_list_val, acc_list, x_axis = [], [], []

    for alpha_th in alpha_ths:
        frac_heads_pruned_list = []
        for _ in range(max_num_batches): 
            x, y = get_batch(P, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(x, targets=y, alpha_th=alpha_th, drop_k=None, get_logits=True)
            nph, nh = outputs['num_head_pruned_per_layer'], outputs['num_heads_per_layer']
            frac_heads_pruned = np.sum(nph) / np.sum(nh) # fractions of heads removed given alpha_th
            frac_heads_pruned_list.append(frac_heads_pruned)
            loss_list_val.append(outputs['ce_loss'])
            acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

        x_axis.append(np.mean(frac_heads_pruned_list))
        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


def save_checkpoint(distributed_backend, model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': distributed_backend.get_raw_model(model).state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)
