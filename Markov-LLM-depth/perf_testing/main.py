import torch
import time

# Unoptimized version of get_batch
def get_batch_unoptimized(P, order, seq_length, batch_size, generator, extra_args):
    data = torch.zeros(batch_size, seq_length + 1, device=extra_args.device)
    alpha = 0.5
    data[:, :order] = torch.bernoulli(alpha * torch.ones((batch_size, order), device=extra_args.device), generator=generator)
    
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(extra_args.device)

    if P is None:
        for b in range(batch_size):
            P = get_random_P(order, generator, extra_args.device, extra_args.dtype)
            for i in range(order, seq_length):
                data[b, i] = get_next_symbols(P, order, data[b, i-order:i])
    else:
        for i in range(order, seq_length):
            prev_symbols = data[:, i-order:i]
            idx = (prev_symbols @ powers).long()
            next_symbols = torch.multinomial(P[idx], 1).squeeze(1)
            data[:, i] = next_symbols

    x = data[:, :seq_length].to(int)
    y = data[:, 1:].to(int)
    return x, y

# Optimized version of get_batch
def get_batch(P, order, seq_length, batch_size, generator, extra_args):
    # Initialize data tensor
    data = torch.zeros(batch_size, seq_length + 1, device=extra_args.device)
    alpha = 0.5
    data[:, :order] = torch.bernoulli(alpha * torch.ones((batch_size, order), device=extra_args.device), generator=generator)
    
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(extra_args.device)

    if P is None:
        # Generate random P for each batch in parallel
        P = get_random_P_batch(order, batch_size, generator, extra_args.device, extra_args.dtype)
        batch_indices = torch.arange(batch_size)
        
        for i in range(order, seq_length):
            # Extract the previous 'order' symbols for the entire batch
            prev_symbols = data[:, i-order:i]

            # Compute indices using the dot product with powers of 2
            idx = (prev_symbols @ powers).long()

            # Fetch next symbols from the transition matrix P for each batch in parallel
            next_symbols = torch.multinomial(P[batch_indices, idx], 1).squeeze(1)

            # Update the data with the newly sampled symbols
            data[:, i] = next_symbols
    else:
        for i in range(order, seq_length):
            prev_symbols = data[:, i-order:i]
            idx = (prev_symbols @ powers).long()
            next_symbols = torch.multinomial(P[idx], 1).squeeze(1)
            data[:, i] = next_symbols

    # Prepare x and y for return
    x = data[:, :seq_length].to(int)
    y = data[:, 1:].to(int)
    return x, y

# Supporting function: generate random P for each batch in parallel
def get_random_P_batch(order, batch_size, generator, device, dtype):
    pk = torch.rand((batch_size, 2**order, 1), generator=generator, dtype=dtype, device=device)
    P = torch.cat([1 - pk, pk], dim=2)  # Concatenate to get transition probabilities for 0 and 1
    return P

# Supporting function: original P generation
def get_random_P(order, generator, device, dtype):
    pk = torch.rand((2**order, 1), generator=generator, dtype=dtype, device=device)
    P = torch.cat([1 - pk, pk], dim=1)
    return P

# Supporting function: get next symbols for a given P and prev_symbols
def get_next_symbols(P, order, prev_symbols):
    powers = torch.Tensor([2**i for i in reversed(range(order))]).to(P.device)
    idx = (prev_symbols @ powers).long()
    return torch.multinomial(P[idx], 1)

# Test environment setup
class ExtraArgs:
    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype

# Performance test runner
def run_performance_test():
    order = 6
    seq_length = 512 
    batch_size = 64
    generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    extra_args = ExtraArgs(device='cpu')  # Change 'cpu' to 'cuda' if running on GPU

    # Run unoptimized version
    start_time = time.time()
    x_unopt, y_unopt = get_batch_unoptimized(None, order, seq_length, batch_size, generator, extra_args)
    unoptimized_time = time.time() - start_time

    # Run optimized version
    generator = torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    start_time = time.time()
    x_opt, y_opt = get_batch(None, order, seq_length, batch_size, generator, extra_args)
    optimized_time = time.time() - start_time

    # Results
    print(f"Unoptimized version took: {unoptimized_time:.6f} seconds")
    print(f"Optimized version took: {optimized_time:.6f} seconds")

    # Sanity check: Ensure both versions produce the same results
    
    # print(x_unopt)
    # print(x_opt)
    assert torch.equal(x_unopt, x_opt), "Mismatch in x between unoptimized and optimized"
    assert torch.equal(y_unopt, y_opt), "Mismatch in y between unoptimized and optimized"
    print("Both versions produce the same results!")

if __name__ == "__main__":
    run_performance_test()
