import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack


def get_random_P(order, n_minibatch, minibatch_size, generator, dist, device, dtype):
    if dist is None:
        pk = torch.rand((n_minibatch, 2**order, 1), generator=generator, dtype=dtype, device=device)
        P = torch.cat([1 - pk, pk], dim=2)
    else:
        P = dist.sample().to(device)

    return torch.repeat_interleave(P, minibatch_size, dim=0)

def empirical_est(x, y, order, beta=1):
    assert x.size(0) == 1
    device = x.device
    x = x.float().squeeze()
    y = y.float().squeeze()
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(device)
    idx = F.conv1d(x.view(1,-1), powers.view(1,1,-1)).squeeze()
    est_vec = []
    for i in range(2**order):
        mask = (idx == i)
        s = y[order-1:][mask]
        s = torch.cat((torch.Tensor([0]).to(device), s[:-1]))
        s = (s.cumsum(0) + beta) / (torch.arange(len(s), device=device) + 2*beta)
        est_vec.append(s)

    return est_vec

def optimal_est(P, order, sequence_length, generator, dist, extra_args):
    x, y = get_batch(P, order, sequence_length, 1, 4096, generator, dist, extra_args)
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(P.device)
    opt_logits = torch.zeros(x.size(0), x.size(1), P.size(1), device=P.device)
    if order > 1:
        opt_logits[:,:order-1,:] = 0.5*torch.ones(x.size(0), order-1, P.size(1), device=P.device)
    for i in range(order-1, x.size(1)):
        idx = x[:,i-order+1:i+1].float() @ powers
        opt_logits[:,i,:] = P[idx.to(int)]
    opt_logits = torch.log(opt_logits)
    opt_loss = F.nll_loss(opt_logits.view(-1, opt_logits.size(-1)), y.view(-1), ignore_index=-1)

    return opt_loss.item()

# Optimized Markov data generation (thank you @cekbote!)
def get_batch(P, order, seq_length, n_minibatch, minibatch_size, generator, dist, extra_args):
    batch_size = n_minibatch * minibatch_size
    data = torch.zeros(batch_size, seq_length+1, device=extra_args.device)
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(extra_args.device)
    if P == None:
        # Generate first k bits
        alpha = 0.5
        data[:, :order] = torch.bernoulli(alpha * torch.ones((batch_size, order), device=extra_args.device), generator=generator)
        # Generate following bits
        P = get_random_P(order, n_minibatch, minibatch_size, generator, dist, extra_args.device, extra_args.dtype)
        batch_indices = torch.arange(batch_size)
        for i in range(order, seq_length+1):
            # Extract the previous 'order' symbols for the entire batch
            prev_symbols = data[:, i-order:i]
            # Compute indices using the dot product with powers of 2
            idx = (prev_symbols @ powers).int()
            # Fetch next symbols from the transition matrix P for each batch in parallel
            next_symbols = torch.multinomial(P[batch_indices, idx], 1, generator=generator).squeeze(1)
            # Update the data with the newly sampled symbols
            data[:, i] = next_symbols
    else:
        if P.dim() == 2:
            # Use same fixed P for all sequences
            # Generate first k bits
            if extra_args.initial == 'steady':
                if P.size(0) == 2:
                    alpha = P[1,0] / (P[0,1] + P[1,0])
                else:
                    alpha = 0.5
            elif extra_args.initial == 'uniform':
                alpha = 0.5
            else:
                alpha = 0.5
            data[:, :order] = torch.bernoulli(alpha * torch.ones((batch_size, order), device=extra_args.device), generator=generator)
            # Generate following bits
            for i in range(order, seq_length+1):
                prev_symbols = data[:, i-order:i]
                idx = (prev_symbols @ powers).int()
                next_symbols = torch.multinomial(P[idx], 1, generator=generator).squeeze(1)
                data[:, i] = next_symbols
        else:
            # Generate first k bits
            alpha = 0.5
            data[:, :order] = torch.bernoulli(alpha * torch.ones((batch_size, order), device=extra_args.device), generator=generator)
            # Generate following bits
            batch_indices = torch.arange(batch_size)
            for i in range(order, seq_length+1):
                # Extract the previous 'order' symbols for the entire batch
                prev_symbols = data[:, i-order:i]
                # Compute indices using the dot product with powers of 2
                idx = (prev_symbols @ powers).int()
                # Fetch next symbols from the transition matrix P for each batch in parallel
                next_symbols = torch.multinomial(P[batch_indices, idx], 1, generator=generator).squeeze(1)
                # Update the data with the newly sampled symbols
                data[:, i] = next_symbols
    x = data[:,:seq_length].to(int)
    y = data[:,1:].to(int)
    
    return x, y


def eval(model, P, order, sequence_length, batch_size, generator, extra_args, max_num_batches=24, ctx=nullcontext()):
    assert model.training == False
    assert P is not None

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches):
        x, y = get_batch(P, order, sequence_length, 1, batch_size, generator, None, extra_args)
        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity


@torch.no_grad()
def eval_att(model, P, order, sequence_length, batch_size, generator, extra_args, device='cpu', ctx=nullcontext()):
    assert model.training == False
    
    x, y = get_batch(P, order, sequence_length, 1, batch_size, generator, None, extra_args)
    with ctx:
        outputs = model(x, targets=y, get_logits=True, get_att=True)
    att_mean = outputs['att_mean']
    att_std = outputs['att_std']

    return att_mean, att_std


@torch.no_grad()
def eval_probs(model, P, order, sequence_length, generator, extra_args, ctx=nullcontext()):
    assert model.training == False
    assert P is not None
    
    x, y = get_batch(P, order, sequence_length, 1, 1, generator, None, extra_args)

    # Get empirical add-beta estimator
    est_vec = empirical_est(x, y, order)

    with ctx:
        outputs = model(x, targets=y, get_logits=True)

    probs = F.softmax(outputs['logits'], dim=-1)
    xb = x[0].float()
    probsb = probs[0, order-1:]
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(extra_args.device)
    idx = torch.Tensor([xb[i:i+order] @ powers for i in range(sequence_length - order + 1)])
    prob_vec = []
    for i in range(2**order):
        vec = probsb[idx == i][:,1] # estimated p
        prob_vec.append(vec)

    return prob_vec, est_vec


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
