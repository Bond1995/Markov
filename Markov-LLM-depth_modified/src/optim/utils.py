import numpy as np
import torch
import torch.nn.functional as F
import math
from torch.distributions.dirichlet import Dirichlet
from contextlib import nullcontext, contextmanager, ExitStack
from typing import Tuple, Dict


def get_random_P(order, batch_size_per_chain, num_chains, generator, dist, device, dtype):
    # if dist is None:
    #     alpha = 0.5 * torch.ones((batch_size, 2**order, 2))
    #     dist = Dirichlet(alpha)
    # P = dist.sample().to(device)

    if dist is None:
        pk = torch.rand((num_chains, 2**order, 1), generator=generator, dtype=dtype, device=device)
        P = torch.cat([1 - pk, pk], dim=2)
    else:
        P = dist.sample().to(device)

    P = P.unsqueeze(1).expand(-1, batch_size_per_chain, *P.shape[1:]).reshape(-1, *P.shape[1:])

    shuffled_P = P[torch.randperm(P.size(0))]
    # shuffled_P = P  
    
    return shuffled_P


def empirical_est(x, y, order, beta=1):
    assert x.size(0) == 1
    device = x.device
    x = x.float().squeeze()
    y = y.float().squeeze()
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(device)
    idx = F.conv1d(x.view(1, -1), powers.view(1, 1, -1)).squeeze()
    est_vec = []
    for i in range(2**order):
        mask = idx == i
        s = y[order - 1 :][mask]
        s = torch.cat((torch.Tensor([0]).to(device), s[:-1]))
        s = (s.cumsum(0) + beta) / (torch.arange(len(s), device=device) + 2 * beta)
        est_vec.append(s)

    return est_vec


def optimal_est(P, order, sequence_length, generator, dist, extra_args):
    x, y = get_batch(P, order, sequence_length, 128, 64, generator, dist, extra_args)
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(P.device)
    opt_logits = torch.zeros(x.size(0), x.size(1), P.size(1), device=P.device)
    if order > 1:
        opt_logits[:, : order - 1, :] = 0.5 * torch.ones(
            x.size(0), order - 1, P.size(1), device=P.device
        )
    for i in range(order - 1, x.size(1)):
        idx = x[:, i - order + 1 : i + 1].float() @ powers
        opt_logits[:, i, :] = P[idx.to(int)]
    opt_logits = torch.log(opt_logits)
    opt_loss = F.nll_loss(
        opt_logits.view(-1, opt_logits.size(-1)), y.view(-1), ignore_index=-1
    )

    return opt_loss


# Optimized Markov data generation (thank you @cekbote!)
def get_batch(P, order, seq_length,  batch_size_per_chain, num_chains, generator, dist, extra_args, return_P=False):
    batch_size = batch_size_per_chain * num_chains
    data = torch.zeros(batch_size, seq_length + 1, device=extra_args.device)
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(
        extra_args.device
    )
    if P == None:
        # Generate first k bits
        alpha = 0.5
        data[:, :order] = torch.bernoulli(
            alpha * torch.ones((batch_size, order), device=extra_args.device),
            generator=generator,
        )
        # Generate following bits
        P = get_random_P(
            order, batch_size_per_chain, num_chains, generator, dist, extra_args.device, extra_args.dtype
        )
        batch_indices = torch.arange(batch_size)
        for i in range(order, seq_length + 1):
            # Extract the previous 'order' symbols for the entire batch
            prev_symbols = data[:, i - order : i]
            # Compute indices using the dot product with powers of 2
            idx = (prev_symbols @ powers).int()
            # Fetch next symbols from the transition matrix P for each batch in parallel
            next_symbols = torch.multinomial(
                P[batch_indices, idx], 1, generator=generator
            ).squeeze(1)
            # Update the data with the newly sampled symbols
            data[:, i] = next_symbols
    else:
        if P.dim() == 2:
            # Use same fixed P for all sequences
            # Generate first k bits
            if extra_args.initial == "steady":
                if P.size(0) == 2:
                    alpha = P[1, 0] / (P[0, 1] + P[1, 0])
                else:
                    alpha = 0.5
            elif extra_args.initial == "uniform":
                alpha = 0.5
            else:
                alpha = 0.5
            data[:, :order] = torch.bernoulli(
                alpha * torch.ones((batch_size, order), device=extra_args.device),
                generator=generator,
            )
            # Generate following bits
            for i in range(order, seq_length + 1):
                prev_symbols = data[:, i - order : i]
                idx = (prev_symbols @ powers).int()
                next_symbols = torch.multinomial(
                    P[idx], 1, generator=generator
                ).squeeze(1)
                data[:, i] = next_symbols
        else:
            # Generate first k bits
            alpha = 0.5
            data[:, :order] = torch.bernoulli(
                alpha * torch.ones((batch_size, order), device=extra_args.device),
                generator=generator,
            )
            # Generate following bits
            batch_indices = torch.arange(batch_size)
            for i in range(order, seq_length + 1):
                # Extract the previous 'order' symbols for the entire batch
                prev_symbols = data[:, i - order : i]
                # Compute indices using the dot product with powers of 2
                idx = (prev_symbols @ powers).int()
                # Fetch next symbols from the transition matrix P for each batch in parallel
                next_symbols = torch.multinomial(
                    P[batch_indices, idx], 1, generator=generator
                ).squeeze(1)
                # Update the data with the newly sampled symbols
                data[:, i] = next_symbols
    x = data[:, :seq_length].to(int)
    y = data[:, 1:].to(int)

    if return_P:
        return x, y, P
    
    return x, y


def get_random_P_old(order, batch_size, generator, device, dtype):
    P = torch.zeros(2**order, 2, dtype=dtype, device=device)
    for k in range(2**order):
        pk = torch.rand(1, generator=generator, dtype=dtype, device=device)
        P[k, :] = torch.Tensor([1 - pk, pk])

    return P


def get_batch_old(P, order, seq_length, batch_size, generator, extra_args):
    data = torch.zeros(batch_size, seq_length + 1, device=extra_args.device)
    if P == None:
        # Generate first k bits
        alpha = 0.5
        for k in range(order):
            data[:, k] = torch.bernoulli(
                alpha * torch.ones((batch_size,), device=extra_args.device),
                generator=generator,
            )
        # Generate following bits
        for b in range(batch_size):
            # New random P for every sequence
            P = get_random_P_old(
                order, 1, generator, extra_args.device, extra_args.dtype
            )
            for i in range(order, seq_length):
                data[b, i] = get_next_symbols(P, order, data[b, i - order : i])
    else:
        # Use same fixed P for all sequences
        # Generate first k bits
        if extra_args.initial == "steady":
            if P.size(0) == 2:
                alpha = P[1, 0] / (P[0, 1] + P[1, 0])
            else:
                alpha = 0.5
        elif extra_args.initial == "uniform":
            alpha = 0.5
        else:
            alpha = 0.5
        for k in range(order):
            data[:, k] = torch.bernoulli(
                alpha * torch.ones((batch_size,), device=extra_args.device),
                generator=generator,
            )
        for i in range(order, seq_length):
            data[:, i] = get_next_symbols(P, order, data[:, i - order : i])
    x = data[:, :seq_length].to(int)
    y = data[:, 1:].to(int)

    return x, y


def get_next_symbols(P, order, data):
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(data.device)
    idx = data @ powers
    M = P[idx.to(int)]
    s = torch.multinomial(M, 1).flatten()

    return s


def eval(
    model,
    P,
    order,
    sequence_length,
    batch_size_per_chain, 
    num_chains,
    generator,
    extra_args,
    max_num_batches=24,
    ctx=nullcontext(),
):
    # assert model.training == False
    assert P is not None

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches):
        x, y = get_batch(
            P, order, sequence_length, batch_size_per_chain, num_chains, generator, None, extra_args
        )
        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs["loss"]
        loss_list_val.append(val_loss)
        acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828**val_loss

    return val_acc, val_loss, val_perplexity


@torch.no_grad()
def eval_att(
    model,
    P,
    order,
    sequence_length,
    batch_size_per_chain, 
    num_chains,
    generator,
    extra_args,
    device="cpu",
    ctx=nullcontext(),
):
    assert model.training == False

    x, y = get_batch(P, order, sequence_length, batch_size_per_chain, num_chains, generator, None, extra_args)
    with ctx:
        outputs = model(x, targets=y, get_logits=True, get_att=True)
    att_mean = outputs["att_mean"]
    att_std = outputs["att_std"]

    return att_mean, att_std


@torch.no_grad()
def eval_probs(
    model, P, order, sequence_length, generator, extra_args, ctx=nullcontext()
):
    assert model.training == False
    assert P is not None

    x, y = get_batch(P, order, sequence_length, 1, 1, generator, None, extra_args)

    # Get empirical add-beta estimator
    est_vec = empirical_est(x, y, order)

    with ctx:
        outputs = model(x, targets=y, get_logits=True)

    probs = F.softmax(outputs["logits"], dim=-1)
    xb = x[0].float()
    probsb = probs[0, order - 1 :]
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(
        extra_args.device
    )
    idx = torch.Tensor(
        [xb[i : i + order] @ powers for i in range(sequence_length - order + 1)]
    )
    prob_vec = []
    for i in range(2**order):
        vec = probsb[idx == i][:, 1]  # estimated p
        prob_vec.append(vec)

    return prob_vec, est_vec


@torch.no_grad()
def eval_sparse(
    model,
    P,
    sequence_length,
    batch_size,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
    alpha_th=None,
    drop_k=None,
):
    assert model.training == False

    ce_loss_list_val, l1_loss_list_val, acc_list, sparcity_per_layer = [], [], [], []

    for _ in range(max_num_batches):
        x, y = get_batch(P, sequence_length, batch_size, device=device)
        with ctx:
            outputs = model(
                x,
                targets=y,
                alpha_th=alpha_th,
                drop_k=drop_k,
                get_logits=True,
                get_alphas=True,
            )
        ce_loss_list_val.append(outputs["ce_loss"])
        l1_loss_list_val.append(outputs["l1_loss"])
        acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())
        sparcity_per_layer.append(
            [
                (alphas < 1e-8).sum().float().cpu().item() / alphas.numel()
                for alphas in outputs["alphas"]
            ]
        )

    val_acc = torch.stack(acc_list).mean().item()
    val_ce_loss = np.mean(ce_loss_list_val)
    val_l1_loss = np.mean(l1_loss_list_val)
    val_perplexity = 2.71828**val_ce_loss
    sparcity_per_layer = np.mean(np.array(sparcity_per_layer), axis=0)

    return val_acc, val_ce_loss, val_l1_loss, val_perplexity, sparcity_per_layer


@torch.no_grad()
def eval_sweep_dropk(
    model,
    P,
    sequence_length,
    batch_size,
    n_heads,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    x_axis, y_axis_pp, y_axis_acc, y_axis_loss = (
        torch.linspace(0.0, 0.95, 15),
        [],
        [],
        [],
    )
    loss_list_val, acc_list = [], []

    for frac in x_axis:
        drop_k = int(sequence_length * frac * n_heads)
        for _ in range(max_num_batches):
            x, y = get_batch(P, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=None, drop_k=drop_k, get_logits=True
                )
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


@torch.no_grad()
def eval_sweep_alphath(
    model,
    P,
    sequence_length,
    batch_size,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    alpha_ths, y_axis_pp, y_axis_acc, y_axis_loss = (
        [0, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1],
        [],
        [],
        [],
    )
    loss_list_val, acc_list, x_axis = [], [], []

    for alpha_th in alpha_ths:
        frac_heads_pruned_list = []
        for _ in range(max_num_batches):
            x, y = get_batch(P, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=alpha_th, drop_k=None, get_logits=True
                )
            nph, nh = (
                outputs["num_head_pruned_per_layer"],
                outputs["num_heads_per_layer"],
            )
            frac_heads_pruned = np.sum(nph) / np.sum(
                nh
            )  # fractions of heads removed given alpha_th
            frac_heads_pruned_list.append(frac_heads_pruned)
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        x_axis.append(np.mean(frac_heads_pruned_list))
        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


def save_checkpoint(
    distributed_backend, model, opt, scheduler, itr, ckpt_path, **extra_args
):
    checkpoint = dict(
        {
            "model": distributed_backend.get_raw_model(model).state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "itr": itr,
        },
        **extra_args,
    )

    torch.save(checkpoint, ckpt_path)


def get_all_transitions_states(order):
    all_states = np.array([i for i in range(2**order)])
    transition_states_0 = np.left_shift(all_states, 1)
    transition_states_1 = np.left_shift(all_states, 1) + 1

    return all_states, transition_states_0, transition_states_1


def get_all_transitions_states(order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate all possible states and their corresponding transitions for a given order.

    Args:
        order (int): The number of bits representing the state.

    Returns:
        Tuple containing:
            - all_states (np.ndarray): Array of all possible states (integers from 0 to 2**order - 1).
            - transition_states_0 (np.ndarray): States transitioned with an appended 0 bit (left-shifted).
            - transition_states_1 (np.ndarray): States transitioned with an appended 1 bit (left-shifted + 1).
    """
    all_states = np.arange(
        2**order
    )  # Optimized by using np.arange instead of list comprehension
    transition_states_0 = np.left_shift(all_states, 1) % 2**order
    transition_states_1 = (transition_states_0 + 1) % 2**order # Don't think we need the 2**order here but there is no harm in keeping it

    return all_states, transition_states_0, transition_states_1


def get_true_transition_states(P: np.ndarray, order: int) -> Dict[str, float]:
    """
    Generate the true transition probabilities for all states based on the transition probability matrix.

    Args:
        P (np.ndarray): Transition probability matrix with shape (2**order, 2).
        order (int): The number of bits representing the state.

    Returns:
        Dict[str, float]: A dictionary where the keys are transition descriptions, and values are probabilities.
    """
    transition_dict = {}
    all_states, transition_states_0, transition_states_1 = get_all_transitions_states(
        order
    )

    for i in range(len(all_states)):
        transition_dict[f"{all_states[i]} -> {transition_states_0[i]}"] = P[
            all_states[i], 0
        ].item()
        transition_dict[f"{all_states[i]} -> {transition_states_1[i]}"] = P[
            all_states[i], 1
        ].item()

    return transition_dict


def estimate_transition_states(
    x: torch.Tensor, logits: torch.Tensor, order: int, burn_period: int = 0
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    """
    Estimate transition states based on logits and actual states from samples, after a burn-in period.

    Args:
        x (torch.Tensor): Tensor of sampled states with shape (batch_size, order).
        logits (torch.Tensor): Tensor of logits representing transition probabilities with shape (batch_size, 2).
        order (int): The number of bits representing the state.
        burn_period (int): The burn-in period to exclude from the analysis.

    Returns:
        Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
            - transition_dict_mean: Mean of the transition logits for each state.
            - transition_dict_std: Standard deviation of the transition logits for each state.
            - transition_dict_count: Count of occurrences for each state.
    """
    # Convert x and logits to numpy arrays after burn-in period
    x = x.cpu().numpy()
    x = np.lib.stride_tricks.sliding_window_view(x, window_shape=order)
    x = x[burn_period:]
    logits = logits.detach().view(-1, 2).cpu().numpy()
    logits = logits[order - 1 :]
    logits = logits[burn_period:]

    all_states, transition_states_0, transition_states_1 = get_all_transitions_states(
        order
    )

    transition_dict_mean = {}
    transition_dict_std = {}
    transition_dict_count = {}
    transition_last_state = {}
    transition_last_state_index = {}

    # Vectorized computation using numpy
    for i in range(len(all_states)):
        mask = x == all_states[i]
        mask = mask.all(axis=1)
        last_index = np.where(mask)[0][-1] if np.any(mask) else None
        if np.any(mask):  # Only calculate if the state exists
            transition_dict_mean[
                f"{all_states[i]} -> {transition_states_0[i]}"
            ] = np.mean(logits[mask, 0])
            transition_dict_mean[
                f"{all_states[i]} -> {transition_states_1[i]}"
            ] = np.mean(logits[mask, 1])

            transition_dict_std[
                f"{all_states[i]} -> {transition_states_0[i]}"
            ] = np.std(logits[mask, 0])
            transition_dict_std[
                f"{all_states[i]} -> {transition_states_1[i]}"
            ] = np.std(logits[mask, 1])

            count = np.sum(mask)
            transition_dict_count[
                f"{all_states[i]} -> {transition_states_0[i]}"
            ] = count
            transition_dict_count[
                f"{all_states[i]} -> {transition_states_1[i]}"
            ] = count
            
            transition_last_state[f"{all_states[i]} -> {transition_states_0[i]}"] = float(logits[last_index, 0])
            transition_last_state[f"{all_states[i]} -> {transition_states_1[i]}"] = float(logits[last_index, 1])
            
            transition_last_state_index[f"{all_states[i]}"] = float(last_index)
            

    return transition_dict_mean, transition_dict_std, transition_dict_count, transition_last_state, transition_last_state_index


def compute_divergences(dict1, dict2, epsilon=1e-10):
    """
    Compute KL divergence (P || Q), symmetric KL divergence, and Total Variation Distance (TVD)
    between two probability distributions represented as dictionaries.

    Parameters:
    - dict1: First probability distribution (P)
    - dict2: Second probability distribution (Q)
    - epsilon: Small constant to handle zero probabilities in log calculations

    Returns:
    - A dictionary containing KL divergence (P || Q), symmetric KL divergence, and TVD
    """

    # Ensure both dicts have the same keys (union of keys)
    keys = set(dict1.keys()).union(set(dict2.keys()))

    kl_pq = 0.0  # KL(P || Q)
    kl_qp = 0.0  # KL(Q || P)
    tvd = 0.0  # Total Variation Distance

    for key in keys:
        p = dict1.get(key, 0.0)
        q = dict2.get(key, 0.0)

        # Compute KL divergence (P || Q)
        if p > 0:
            kl_pq += p * math.log(p / (q + epsilon))

        # Compute KL divergence (Q || P)
        if q > 0:
            kl_qp += q * math.log(q / (p + epsilon))

        # Compute Total Variation Distance (TVD)
        tvd += abs(p - q)

    # KL Divergence (P || Q) and Symmetric KL
    kl_symmetric = kl_pq + kl_qp
    tvd = tvd / 2.0  # Total variation distance is half the sum of absolute differences

    return {
        "kl": kl_pq,
        "sym_kl": kl_symmetric,
        "tvd": tvd,
    }
