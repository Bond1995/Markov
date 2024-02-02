import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack


def get_batch(P, seq_length, batch_size, generator, extra_args, device='cpu'):
    data = torch.zeros(batch_size, seq_length+1, device=device)
    if extra_args.initial == 'steady':
        alpha = P[0,1]/(P[0,1]+P[1,0])
    elif extra_args.initial == 'uniform':
        alpha = 0.5
    else:
        alpha = 0.5
    data[:,0] = torch.bernoulli(alpha*torch.ones((batch_size,), device=device), generator=generator)
    for i in range(seq_length):
        data[:,i+1] = get_next_symbols(P, data[:,i])
    x = data[:,:seq_length].to(int)
    y = data[:,1:].to(int)
    #if "cuda" in torch.device(device).type:
    #    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    #    x = x.pin_memory().to(device, non_blocking=True)
    #    y = y.pin_memory().to(device, non_blocking=True)
    return x, y

def get_next_symbols(P, data):
    M = P[data.to(int)]
    s = torch.multinomial(M,1).flatten()

    return s

def DKL(P, Q):
    d = 0
    for i in range(P.size(dim=0)):
        d += P[i] * torch.log(P[i]/Q[i])
    
    return max(d, 0.0)

def baseline_est(x):
    if x[-1] == 1:
        if x.sum() == x.numel() - torch.argmax(x):
            l = x.sum()
            if l > x.numel()/2:
                p = 1.0 - 1.0 / l
            else:
                p = 1.0 - 1.0 / (l * np.log(x.numel()))
        else:
            y = (x[:-1] == 1)
            z = x[1:][y]
            p = (z.sum()+0.5) / (z.numel()+1)
    elif x[-1] == 0:
        if x.sum() == torch.argmin(x):
            l = x.numel() - x.sum()
            if l > x.numel()/2:
                p = 1.0 / l
            else:
                p = 1.0 / (l * np.log(x.numel()))
        else:
            y = (x[:-1] == 0)
            z = x[1:][y]
            p = (z.sum()+0.5) / (z.numel()+1)
    else:
        raise ValueError("Error with estimator baseline!")
    
    return torch.tensor([1-p, p])


@torch.no_grad()
def eval(model, P, sequence_length, batch_size, generator, extra_args, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches): 
        x, y = get_batch(P, sequence_length, batch_size, generator, extra_args, device=device)
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
def eval_probs(model, P, sequence_length, names, generator, extra_args, device='cpu', ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    x, y = get_batch(P, sequence_length, len(names), generator, extra_args, device=device)
    with ctx:
        outputs = model(x, targets=y, get_logits=True)
    val_loss = outputs['loss']
    loss_list_val.append(val_loss)
    acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    probs = F.softmax(outputs['logits'], dim=-1)

    prob_dict = dict()
    for i, name in enumerate(names):
        xb = x[i].to(bool)
        probsb = probs[i]
        vec1 = probsb[xb][:,0]
        vec2 = probsb[torch.logical_not(xb)][:,0]
        prob_dict[name] = [vec1, vec2]

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity, prob_dict

@torch.no_grad()
def eval_baseline(model, est, P, sequence_length, batch_size, iterations, generator, extra_args, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches): 
        x, y = get_batch(P, sequence_length, batch_size, generator, extra_args, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    probs = F.softmax(outputs['logits'], dim=-1)

    M = torch.zeros_like(P, device=device)
    M[0] = probs[x==0].mean(dim=0)
    M[1] = probs[x==1].mean(dim=0)
    E = est.estimate()
    p = P[0,1]
    q = P[1,0]
    est_loss = (q*torch.linalg.norm(M[0] - P[0]) + p*torch.linalg.norm(M[1]-P[1])) / (p+q)
    baseline_est_loss = (q*torch.linalg.norm(E[0] - P[0]) + p*torch.linalg.norm(E[1]-P[1])) / (p+q)

    pred_loss = 0
    #baseline_loss = 0
    baseline_loss_history = 0

    #x2, y2 = get_batch(P, sequence_length*iterations, batch_size, device=device)

    for b in range(batch_size):
        #M[b,x[b,i-1],x[b,i]] += 1
        pred_loss += DKL(P[x[b,-1]],probs[b,-1]) / batch_size
        #baseline_loss += DKL(P[x2[b,-1]],baseline_est(x2[b])) / batch_size
        baseline_loss_history += DKL(P[x[b,-1]],E[x[b,-1]]) / batch_size

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity, pred_loss, baseline_loss_history, est_loss, baseline_est_loss

@torch.no_grad()
def eval_pred_baseline(P, sequence_length, batch_size, iterations, generator, extra_args, device='cpu'):
    baseline_loss = 0
    x2, y2 = get_batch(P, sequence_length*iterations, batch_size, generator, extra_args, device=device)

    for b in range(batch_size):
        baseline_loss += DKL(P[x2[b,-1]],baseline_est(x2[b])) / batch_size

    return baseline_loss

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
