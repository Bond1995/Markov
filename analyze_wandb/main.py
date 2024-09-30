import os
import numpy as np
import math
from utils import list_subfolders, load_json, load_yaml, get_sorted_files, get_all_diagonals, get_order_diagonals, compute_mean_std, write_to_json
from plots import plot_mean_std, plot_and_save_heatmap, create_bar_plots_in_subplots


def save_attention_plots(wandb_folder_path, wandb_folder_name, save_folder_name, n_layer, sample_size, order):
    sorted_attention_files = get_sorted_files(os.path.join(wandb_folder_path, wandb_folder_name, f"files/att-id{i}/weights"))
    num_files = len(sorted_attention_files)
    interval = math.ceil(num_files / sample_size)
    sampled_files = [sorted_attention_files[i] for i in range(0, num_files, interval)] + [sorted_attention_files[-1]]
    for file_name in sampled_files:
        weight = np.load(os.path.join(wandb_folder_path, wandb_folder_name, f"files/att-id{i}/weights", file_name))
        
        plot_and_save_heatmap(weight, os.path.join(save_folder_name, file_name.replace(".npy", ".png")))
        
        stats_order_file_name = file_name.replace(".npy", "") + "_stats_order.png"
        means, stds = compute_mean_std(get_order_diagonals(weight, order))
        plot_mean_std(means, stds, order, os.path.join(save_folder_name, stats_order_file_name))
        print(f"Saved plot for {file_name}")
        
        stats_all_file_name = file_name.replace(".npy", "") + "_stats.png"
        means, stds = compute_mean_std(get_all_diagonals(weight))
        plot_mean_std(means, stds, weight.shape[0], os.path.join(save_folder_name, stats_all_file_name))
        print(f"Saved plot for {file_name}")
        
            

if __name__ == "__main__":
    wandb_folder_path = "/mlbio_scratch/ekbote/Markov/Markov-LLM-depth/wandb"
    wandb_folders = list_subfolders(wandb_folder_path)
    threshold = 0.03
    save_folder_path = "optimal"
    sample_size = 20
    
    
    for wandb_folder_name in wandb_folders:
        try:
            wandb_summary = load_json(os.path.join(wandb_folder_path, wandb_folder_name, "files/wandb-summary.json"))
            wandb_config = load_yaml(os.path.join(wandb_folder_path,  wandb_folder_name, "files/config.yaml"))
            val_loss = wandb_summary["val/loss"]
            val_opt_loss = wandb_summary["val/opt_loss"]
            print(wandb_config)
            if np.abs(val_loss - val_opt_loss) <= threshold:
                
                n_layer = wandb_config["n_layer"]["value"]
                n_embd = wandb_config["n_embd"]["value"]
                order = wandb_config["order"]["value"]
                seed = wandb_config["seed"]["value"]
                save_folder_name = f"{save_folder_path}/n_layer-{n_layer}/n_order-{order}/n_embd-{n_embd}/n_seed-{seed}"
                os.makedirs(save_folder_name, exist_ok=True)
                
                optimal_data = {
                    "val_loss": val_loss,
                    "val_opt_loss": val_opt_loss
                }
                
                write_to_json(optimal_data, os.path.join(save_folder_name, "loss_comparision.json"))
                
                
                rank_information = {}
                for i in range(n_layer):
                    rank_information[f"q-id{i}"] = [wandb_summary[f"q-id{i}-energy{j+1}"] for j in range(5)]
                    rank_information[f"k-id{i}"] = [wandb_summary[f"k-id{i}-energy{j+1}"] for j in range(5)]
                    rank_information[f"v-id{i}"] = [wandb_summary[f"v-id{i}-energy{j+1}"] for j in range(5)]
                plot_filename = os.path.join(save_folder_name, "rank_information_plot.png")
                create_bar_plots_in_subplots(rank_information, n_layer, plot_filename)
                # Create a figure with 3 subplots per layer (one for q-id, k-id, and v-id)
                    
                
                for i in range(n_layer):
                    # Saving Atteion
                    sorted_attention_files = get_sorted_files(os.path.join(wandb_folder_path, wandb_folder_name, f"files/att-id{i}/weights"))
                    save_attention_folder = os.path.join(save_folder_name, f"att-id-{i}")
                    os.makedirs(save_attention_folder, exist_ok=True)
                    save_attention_plots(wandb_folder_path, wandb_folder_name, save_attention_folder, n_layer, sample_size, order)
                

        except Exception as e:
            print(f"Error loading files for {wandb_folder_name}: {e}")
            continue