from contextlib import nullcontext

import torch
import torch.nn.functional as F
import wandb
import time 
import copy

from .utils import eval, eval_probs, eval_att, get_batch, save_checkpoint


def train_base(model, opt, P, order, scheduler, iterations, acc_steps, batch_size, sequence_length, generator, eval_freq, ckpt_path, distributed_backend, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.float16)  # extra_args.dtype) #changed!
    itr, substep, best_val_loss, text_table = 0, 0, float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 

    stats = {'train_loss': [], 'val_loss': [], 'val_pp': [], 'val_acc': []}
    
    if not extra_args.no_compile:
        print(f"Compiling model ...")
        model = torch.compile(model) # requires pytorch 2.0+

    print("Markov transition matrix:")
    print(P)

    model.train()

    t0 = time.time()
    while itr < iterations:
        for microstep_idx in range(acc_steps):  # gradient accumulation
            x, y = get_batch(P, order, sequence_length, batch_size, generator, extra_args, device=extra_args.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                    outputs = model(x, targets=y)
            loss = outputs['loss'] / acc_steps
            loss.backward()
            substep += 1

        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
            if distributed_backend.is_master_process():
                t1 = time.time()
                dt = t1 - t0

                model.eval()
                train_loss = loss.detach().cpu().item()
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
                val_acc, val_loss, val_perplexity = eval(model, P, order, sequence_length, batch_size,
                                                        generator, extra_args, device=extra_args.device, max_num_batches=10, ctx=type_ctx)

                print_string = f"{itr} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
                print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
                if scheduler is not None:
                    print_string += f" [lr] {current_lr:.5f}"
                print(print_string)

                if extra_args.wandb:
                    wandb.log({
                        "iter": itr,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/perplexity": val_perplexity,
                        "val/acc": val_acc,
                        "lr": current_lr,
                    })
                
                if itr == iterations:
                    _, _, _, prob_vec, opt_loss = eval_probs(model, P, order, sequence_length, generator, extra_args,
                                                         device=extra_args.device, ctx=type_ctx)
                    if extra_args.wandb:
                        for k in range(2**order):
                            for i in range(len(prob_vec[k])):
                                wandb.log({
                                    "est/est_" + str(k): prob_vec[k][i].detach().cpu().item(),
                                })

                    if extra_args.wandb:
                        wandb.log({
                            "val/opt_loss": opt_loss.detach().cpu().item(),
                        })

                    att_mean, att_std = eval_att(model, P, order, sequence_length, 100,
                                                        generator, extra_args, device=extra_args.device, ctx=type_ctx)
                    if extra_args.wandb:
                        wandb.log({
                            "val/att_mean": att_mean,
                            "val/att_std": att_std,
                        })
                    
                    # if extra_args.eval_seq_prefix != 'none' and (itr % (eval_freq * 5) == 0 or itr == iterations):
                    #     if text_table is None:
                    #         text_table = wandb.Table(columns=["itr", "val-pp", "text"])

                    #     out_str = distributed_backend.get_raw_model(model).generate_from_string(
                    #         extra_args.eval_seq_prefix, max_new_tokens=40, temperature=0.9, top_k=None)
                    #     text_table.add_data(itr, val_perplexity, out_str)
                    #     # why a copy? see github.com/wandb/wandb/issues/2981
                    #     wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})

                model.train()
                t0 = time.time()

    if distributed_backend.is_master_process():
        print(f"saving checkpoint to {ckpt_path}")
        save_checkpoint(distributed_backend=distributed_backend,
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
                        itr=itr,
                        ckpt_path=ckpt_path)

    return stats

