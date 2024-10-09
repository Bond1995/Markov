from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import wandb
import time 
import copy

from .utils import get_random_P, optimal_est, eval, eval_probs, eval_att, get_batch, save_checkpoint
from .utils import get_true_transition_states, estimate_transition_states, compute_divergences
import numpy as np
import json
import os
import json


def set_eval_mode(model):
    # Disable gradients
    model.eval()  # Set model to evaluation mode (affects dropout, batchnorm, etc.)
    for param in model.parameters():
        param.requires_grad = False  # Disable gradient calculation for all parameters
    torch.set_grad_enabled(False)  # Globally disable gradients
    
    
def set_train_mode(model):
    # Enable gradients
    model.train()  # Set model to training mode (re-enables dropout, batchnorm, etc.)
    for param in model.parameters():
        param.requires_grad = True  # Enable gradient calculation for all parameters
    torch.set_grad_enabled(True)  # Globally enable gradients

metrics_list = []

def save_metrics_to_json(metrics, filename="metrics.json"):
    # Function to recursively convert Tensors to lists in the dictionary
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()  # Convert tensor to list
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_list(i) for i in obj]
        else:
            return obj

    # Convert the metrics to a JSON-serializable format
    serializable_metrics = tensor_to_list(metrics)

    # Save the list of metrics to a JSON file
    with open(filename, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)


def train_base(model, opt, P, order, scheduler, iterations, acc_steps, batch_size_per_chain, num_chains, sequence_length, generator, eval_freq, ckpt_path, distributed_backend, extra_args):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.float16)  # extra_args.dtype) #changed!
    itr, substep, best_val_loss, text_table = 0, 0, float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 

    stats = {'train_loss': [], 'val_loss': [], 'val_pp': [], 'val_acc': []}
    batch_size = batch_size_per_chain * num_chains
    
    if not extra_args.no_compile:
        print(f"Compiling model ...")
        model = torch.compile(model) # requires pytorch 2.0+

    alpha = 0.5 * torch.ones((batch_size, 2**order, 2))
    dist = Dirichlet(alpha)
    dist = None

    if P is not None:
        P_test = P
        print("Markov transition matrix:")
        print(P)
    else:
        P_test = get_random_P(order, 1, 1, generator, None, extra_args.device, extra_args.dtype).squeeze(0)
        print("Test Markov transition matrix:")
        print(P_test)
    
    # Optimal test loss
    opt_loss = optimal_est(P_test, order, sequence_length, generator, dist, extra_args)
    if extra_args.wandb:
        wandb.log({
            "val/opt_loss": opt_loss,
        })

    
    set_train_mode(model)
    
    t0 = time.time()
    while itr < iterations:
        
        model.train()
        
        for microstep_idx in range(acc_steps):  # gradient accumulation
            x, y, P_batch = get_batch(None, order, sequence_length, batch_size_per_chain, num_chains, generator, dist, extra_args, return_P=True)
            with type_ctx:
                outputs = model(x, targets=y, )
            loss = outputs['loss'] / acc_steps
            loss.backward()
            substep += 1
            
        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1
        print(f"Training iteration {itr} | Loss: {loss.item()}")
        
        if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
            
            # set_eval_mode(model)
            model.eval()
            
            # Estimating the transition states
            x, y, P_batch = get_batch(None, order, sequence_length, batch_size_per_chain, num_chains, generator, dist, extra_args, return_P=True)
            
            outputs = model(x, targets=y, get_logits=True)
            
            kl, sym_kl, tvd = [], [], []
            kl_last_transition, sym_kl_last_transition, tvd_last_transition = [], [], []
            for i in range(batch_size):
                true_P_dict = get_true_transition_states(P_batch[i], order)
                est_P_dict_mean, est_P_dict_std, est_P_dict_count, est_last_transition_prob, est_last_transition_index = estimate_transition_states(x=x[i], order=order, logits=outputs["logits"][i])
                divs = compute_divergences(true_P_dict, est_P_dict_mean)
                kl.append(divs["kl"])
                sym_kl.append(divs["sym_kl"])
                tvd.append(divs["tvd"])
                
                divs_last_transition = compute_divergences(true_P_dict, est_last_transition_prob)
                kl_last_transition.append(divs_last_transition["kl"])
                sym_kl_last_transition.append(divs_last_transition["sym_kl"])
                tvd_last_transition.append(divs_last_transition["tvd"])
                
                
                if i == batch_size-1:
                    transition_path = f"{model.ckpt_path}/transition_states"
                    true_p_path = f"{transition_path}/true-P"
                    os.makedirs(true_p_path, exist_ok=True)
                    with open(f"{true_p_path}/true-P-{itr}.json", 'w') as json_file:
                        json.dump(true_P_dict, json_file, indent=4)  # indent for pretty-printing

                    est_p_mean_path = f"{transition_path}/est-P-mean"
                    est_P_dict_mean = {k: float(v) for k, v in est_P_dict_mean.items()}
                    os.makedirs(est_p_mean_path, exist_ok=True)
                    with open(f"{est_p_mean_path}/est-P-mean-{itr}.json", 'w') as json_file:
                        json.dump(est_P_dict_mean, json_file, indent=4)  # indent for pretty-printing
                    
                    est_P_std_path = f"{transition_path}/est-P-std"
                    est_P_dict_std = {k: float(v) for k, v in est_P_dict_std.items()}
                    os.makedirs(est_P_std_path, exist_ok=True)
                    with open(f"{est_P_std_path}/est-P-std-{itr}.json", 'w') as json_file:
                        json.dump(est_P_dict_std, json_file, indent=4)  # indent for pretty-printing
                    
                    est_P_count_path = f"{transition_path}/est-P-count"
                    est_P_dict_count = {k: float(v) for k, v in est_P_dict_count.items()}
                    os.makedirs(est_P_count_path, exist_ok=True)
                    with open(f"{est_P_count_path}/est-P-count-{itr}.json", 'w') as json_file:
                        json.dump(est_P_dict_count, json_file, indent=4)  # indent for pretty-printing
                        
                    est_last_transition_prob_path = f"{transition_path}/est-transition-prob"
                    os.makedirs(est_last_transition_prob_path, exist_ok=True)
                    
                    with open(f"{est_last_transition_prob_path}/est-transition-prob-{itr}.json", 'w') as json_file:
                        json.dump(est_last_transition_prob, json_file, indent=4)
                    
                    with open(f"{est_last_transition_prob_path}/est-transition-index-{itr}.json", 'w') as json_file:
                        json.dump(est_last_transition_index, json_file, indent=4)                   
                
                
                #TODO: Save the estimated transition states and the true transition states for the last state of each batch
            
            if extra_args.wandb:
                wandb.log({
                    "prob_est/kl_mean": np.mean(kl),
                    "prob_est/sym_kl_mean": np.mean(sym_kl),
                    "prob_est/tvd_mean": np.mean(tvd),
                    "prob_est/kl_std": np.std(kl),
                    "prob_est/sym_kl_std": np.std(sym_kl),
                    "prob_est/tvd_std": np.std(tvd),
                    "prob_est_last/kl_last_transition_mean": np.mean(kl_last_transition),
                    "prob_est_last/sym_kl_last_transition_mean": np.mean(sym_kl_last_transition),
                    "prob_est_last/tvd_last_transition_mean": np.mean(tvd_last_transition),
                    "prob_est_last/kl_last_transition_std": np.std(kl_last_transition),
                    "prob_est_last/sym_kl_last_transition_std": np.std(sym_kl_last_transition),
                    "prob_est_last/tvd_last_transition_std": np.std(tvd_last_transition)
                })
                
            
            print("Divergences:")
            print(divs)
            
            t1 = time.time()
            dt = t1 - t0

            
            # Conventional Metrics
            
            train_loss = loss.detach().cpu().item()
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
            val_acc, val_loss, val_perplexity = eval(model, P_test, order, sequence_length, batch_size_per_chain, num_chains,
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
                "optimal": True if (abs(val_loss-opt_loss)<0.03) else False,
                "kl_mean": np.mean(kl),
                "sym_kl_mean": np.mean(sym_kl),
                "tvd_mean": np.mean(tvd),
                "kl_std": np.std(kl),
                "sym_kl_std": np.std(sym_kl),
                "tvd_std": np.std(tvd),
                "kl_mean_last_transition": np.mean(kl_last_transition),
                "sym_kl_mean_last_transition": np.mean(sym_kl_last_transition),
                "tvd_mean_last_transition": np.mean(tvd_last_transition),
                "kl_std_last_transition": np.std(kl_last_transition),
                "sym_kl_std_last_transition": np.std(sym_kl_last_transition),
                "tvd_std_last_transition": np.std(tvd_last_transition),
            })
            
            if itr == iterations:
                prob_vec, est_vec = eval_probs(model, P_test, order, sequence_length, generator, extra_args,
                                                        ctx=type_ctx)
                if extra_args.wandb:
                    for k in range(2**order):
                        for i in range(len(prob_vec[k])):
                            wandb.log({
                                "est/model_est_" + str(k): prob_vec[k][i].detach().cpu().item(),
                                "est/empirical_est_" + str(k): est_vec[k][i].detach().cpu().item(),
                            })
                
                att_mean, att_std = eval_att(model, P_test, order, sequence_length, batch_size_per_chain, num_chains,
                                                        generator, extra_args, device=extra_args.device, ctx=type_ctx)
                if extra_args.wandb:
                    wandb.log({
                        "val/att_mean": att_mean,
                        "val/att_std": att_std,
                    })

            # set_train_mode(model)
            
            model.train()
            
            t0 = time.time()
    
    save_metrics_to_json(metrics_list, filename=f"{model.ckpt_path}/metrics.json")

    for i in range(10):
        folder_name = f"post_training-{i}"
        wandb_run_dir = wandb.run.dir
        ckpt_path_per_folder = f"{model.ckpt_path}/{folder_name}"
        print(f"Saving data to {ckpt_path_per_folder}")
        os.makedirs(ckpt_path_per_folder, exist_ok=True)
        alpha = 0.5 * torch.ones((1, 2**order, 2))
        dist = Dirichlet(alpha)
        dist = None
        P_test_i = get_random_P(order, 1, 1, generator, dist, extra_args.device, extra_args.dtype)
        x, y = get_batch(P_test_i, order, sequence_length, 1, 1, generator, dist, extra_args)
        
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
        # try:
        #     artifact_name_est_vec = f"{folder_name}_est_vec"
        #     artifact_est_vec = wandb.Artifact(artifact_name_est_vec, type="dataset")
        #     artifact_est_vec.add_file(f'{ckpt_path_per_folder}/est_vec.pth')
        #     wandb.log({artifact_name_est_vec: artifact_est_vec})
        # except Exception as e:
        #     print(f"Failed to log est_vec image - {folder_name}, Error: {e}")
    
    save_path = f"{model.ckpt_path}/model.pth"
    
    
    print(f"saving checkpoint to {save_path}")

    if distributed_backend.is_master_process():
        print(f"saving checkpoint to {ckpt_path}")
        save_checkpoint(distributed_backend=distributed_backend,
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
                        itr=itr,
                        ckpt_path=save_path,)

    return stats


