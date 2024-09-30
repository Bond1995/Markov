from contextlib import nullcontext
import os

import torch
import torch.nn.functional as F
import wandb
import time 
import time 
import copy
import time
import copy
import numpy as np
import json

from .utils import eval, eval_probs, get_batch, get_random_P, optimal_est, save_checkpoint, get_batch_optimised

metrics_list = []

def save_metrics_to_json(metrics, filename="metrics.json"):
    # Save the list of metrics to a JSON file
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)

def empirical_est(x, y, order, beta=0.5):
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

def train_base(model, opt, P, order, scheduler, iterations, acc_steps, batch_size, sequence_length, generator, eval_freq, ckpt_path, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.float16)
    itr, substep, best_val_loss, text_table = 0, 0, float('inf'), None
    
    print(f"Compiling model ...")
    # model = torch.compile(model) # requires pytorch 2.0+

    if P is not None:
        P_test = P
        print("Markov transition matrix:")
        print(P)
    else:
        P_test = get_random_P(order, generator, extra_args.device, extra_args.dtype)
        print("Test Markov transition matrix:")
        print(P_test)
    
    # Optimal test loss
    opt_loss = optimal_est(P_test, order, sequence_length, generator, extra_args)
    if extra_args.wandb:
        wandb.log({
            "val/opt_loss": opt_loss,
        })

    model.train()
    t0 = time.time()

    print("Total iterations: ", iterations)
    print("Accumulation steps: ", acc_steps)
    print("Batch size: ", batch_size)
    print()
    
    iterations = 501
    
    while itr < iterations:
        for microstep_idx in range(acc_steps):  # gradient accumulation
            P=None 
            x, y = get_batch_optimised(P, order, sequence_length, batch_size, generator, extra_args)
            with type_ctx:
                outputs = model(x, targets=y)
            loss = outputs['loss'] / acc_steps
            loss.backward()
            substep += 1

        print("Iter: ", itr, "Loss: ", loss.item())

        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        if itr % eval_freq == 0 or itr == iterations or itr == 1: # from here it's only evaluation code, all the training is above
            t1 = time.time()
            dt = t1 - t0

            model.eval()
            train_loss = loss.detach().cpu().item()
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
            val_acc, val_loss, val_perplexity = eval(model, P_test, order, sequence_length, batch_size,
                                                    generator, extra_args, max_num_batches=10, ctx=type_ctx)

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

            metrics_list.append({
                "iter": itr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
                "val_acc": val_acc,
                "lr": current_lr,
                "opt_loss": opt_loss,
                "optimal": True if (abs(val_loss-opt_loss)<0.03) else False
            })
            
            # if itr == iterations:
            if itr == iterations or itr == 1:
                prob_vec = eval_probs(model, P_test, order, sequence_length, generator, extra_args,
                                                        ctx=type_ctx)
                if extra_args.wandb:
                    for k in range(2**order):
                        for i in range(len(prob_vec[k])):
                            wandb.log({
                                "est/est_" + str(k): prob_vec[k][i].detach().cpu().item(),
                            })

                    if extra_args.wandb:
                        wandb.log({
                            "val/opt_loss": opt_loss,
                        })
                    
                    # att_mean, att_std = eval_att(model, P, order, sequence_length, 100,
                    #                                     generator, extra_args, device=extra_args.device, ctx=type_ctx)
                    # if extra_args.wandb:
                    #     wandb.log({
                    #         "val/att_mean": att_mean,
                    #         "val/att_std": att_std,
                    #     })
                    
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
        model.update_iter()
    
    save_metrics_to_json(metrics_list, filename=f"{model.ckpt_path}/metrics.json")
    
    for i in range(10):
        folder_name = f"post_training-{i}"
        wandb_run_dir = wandb.run.dir
        ckpt_path_per_folder = f"{model.ckpt_path}/{folder_name}"
        print(f"Saving data to {ckpt_path_per_folder}")
        os.makedirs(ckpt_path_per_folder, exist_ok=True)
        P_test_i = get_random_P(order, generator, extra_args.device, extra_args.dtype)
        x, y = get_batch(P_test_i, order, sequence_length, 1, generator, extra_args)
        
        np.save(f"{ckpt_path_per_folder}/P_test.npy", P_test_i.cpu().numpy())
        # try:
        #     artifact_name_P_test = f"{folder_name}_P_test"
        #     artifact_P_test = wandb.Artifact(artifact_name_P_test, type="dataset")
        #     artifact_P_test.add_file(f'{ckpt_path_per_folder}/P_test.npy')
        #     wandb.log({artifact_name_P_test: artifact_P_test})
        # except Exception as e:
        #     print(f"Failed to log P_test image - {folder_name}, Error: {e}")

        # Save x.npy
        np.save(f"{ckpt_path_per_folder}/x.npy", x.cpu().numpy())
        # try:
        #     artifact_name_x = f"{folder_name}_x"
        #     artifact_x = wandb.Artifact(artifact_name_x, type="dataset")
        #     artifact_x.add_file(f'{ckpt_path_per_folder}/x.npy')
        #     wandb.log({artifact_name_x: artifact_x})
        # except Exception as e:
        #     print(f"Failed to log x image - {folder_name}, Error: {e}")

        
        np.save(f"{ckpt_path_per_folder}/y.npy", y.cpu().numpy())
        outputs = model(x, targets=y, folder_name=folder_name, save_forward=True)
        est_vec = empirical_est(x, y, order, beta=0.5)
        torch.save(est_vec, f"{ckpt_path_per_folder}/est_vec.pth")
        # try:
        #     artifact_name_est_vec = f"{folder_name}_est_vec"
        #     artifact_est_vec = wandb.Artifact(artifact_name_est_vec, type="dataset")
        #     artifact_est_vec.add_file(f'{ckpt_path_per_folder}/est_vec.pth')
        #     wandb.log({artifact_name_est_vec: artifact_est_vec})
        # except Exception as e:
        #     print(f"Failed to log est_vec image - {folder_name}, Error: {e}")
    
    save_path = f"{model.ckpt_path}/model.pth"
    
    
    print(f"saving checkpoint to {save_path}")
    save_checkpoint(model=model,
                    opt=opt,
                    scheduler=scheduler,
                    itr=itr,
                    ckpt_path=save_path)
    
    # Save and log model file artifact
    # try:
    #     artifact_name_model = "model_file"
    #     artifact_model = wandb.Artifact(artifact_name_model, type="model")
    #     artifact_model.add_file(f"{wandb_run_dir}/model.pth")
    #     wandb.log({artifact_name_model: artifact_model})
    # except Exception as e:
    #     print(f"Failed to log model file, Error: {e}")

    wandb.finish()

    
    wandb.finish()

