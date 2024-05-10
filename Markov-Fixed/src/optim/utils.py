import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack


def get_batch(p, q, order, seq_length, batch_size, generator, extra_args, device='cpu'):
    data = torch.zeros(batch_size, seq_length+1, device=device)
    if extra_args.initial == 'steady':
        alpha = q / (p+q)
    elif extra_args.initial == 'uniform':
        alpha = 0.5
    else:
        alpha = 0.5
    # Generate first k bits
    for k in range(order):
        data[:,k] = torch.bernoulli(alpha*torch.ones((batch_size,), device=device), generator=generator)
    for i in range(order, seq_length):
        data[:,i] = get_next_symbols(p, q, data[:,i-order])
    x = data[:,:seq_length].to(int)
    y = data[:,1:].to(int)
    #if "cuda" in torch.device(device).type:
    #    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    #    x = x.pin_memory().to(device, non_blocking=True)
    #    y = y.pin_memory().to(device, non_blocking=True)
    return x, y

def get_next_symbols(p, q, data):
    P = torch.Tensor([[1-p, p],[q, 1-q]]).to(data.device)
    M = P[data.to(int)]
    s = torch.multinomial(M,1).flatten()

    return s


@torch.no_grad()
def eval(model, p, q, order, sequence_length, batch_size, generator, extra_args, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches): 
        x, y = get_batch(p, q, order, sequence_length, batch_size, generator, extra_args, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
        acc_list.append(((outputs['logits'] > 0) == y.to(bool)).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity

@torch.no_grad()
def eval_probs(model, p, q, order, sequence_length, generator, extra_args, device='cpu', ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    x, y = get_batch(p, q, order, sequence_length, 1, generator, extra_args, device=device)
    with ctx:
        outputs = model (x, targets=y, get_logits=True)
    val_loss = outputs['loss']
    loss_list_val.append(val_loss)
    acc_list.append(((outputs['logits'] > 0) == y.to(bool)).float().mean())

    probs = torch.sigmoid(outputs['logits'])

    xb = x[0]
    probsb = probs[0, order-1:]
    if order > 1:
        idx = xb[:-order+1]
    else:
        idx = xb[:]
    vec0 = probsb[idx == 0] # estimated p
    vec1 = 1 - probsb[idx == 1] # estimated q
    prob_vec = [vec0, vec1]

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity, prob_vec


def save_checkpoint(model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)
