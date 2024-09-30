from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def plot_mean_std(means, stds, k, name):
    """
    Plots the mean and standard deviation for each diagonal using bar plots, 
    with the order of diagonals reversed and improved formatting.

    Parameters:
    - means: List of means.
    - stds: List of standard deviations.
    - k: Number of diagonals to label on the x-axis.
    - name: The filename to save the plot.
    """
    # Reverse the order of means, stds, and labels
    means = means[::-1]
    stds = stds[::-1]
    labels = ([f"0"] + [f"-{i}" for i in range(1, k)])[::-1]  # Reverse the labels as well

    # Generate a color gradient for the bars
    cmap = cm.get_cmap('coolwarm', k)
    colors = cmap(np.linspace(0, 1, k))

    plt.figure(figsize=(20, 6))  # Increase figure size to give more horizontal space

    # Create bar plot for means with error bars (standard deviations)
    bar_positions = np.arange(k) * 2  # Create more space between bars by multiplying the positions
    bars = plt.bar(bar_positions, means, yerr=stds, capsize=8, color=colors, edgecolor='black', width=0.8)

    # Set plot labels and title with improved formatting
    plt.xticks(bar_positions, labels, fontsize=12)
    plt.xlabel('Diagonal Index (0: Main, -1: Below Main, etc.)', fontsize=14)
    plt.ylabel('Mean and Standard Deviation', fontsize=14)
    plt.title('Mean and Standard Deviation of Diagonals (Reversed Order)', fontsize=16, pad=15)

    # Adjust x-axis limits to give more spacing around the bars
    plt.xlim(-1, max(bar_positions) + 1)

    # Add grid for better readability
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

    # Save the plot
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    plt.close()





    
def plot_and_save_heatmap(data, filename='heatmap.png', cmap='viridis', dpi=300):
    """
    Plots a heatmap using imshow and saves it as an image file.

    Parameters:
    - data: 2D array-like data to be plotted in the heatmap.
    - filename: The name of the file to save the heatmap (default 'heatmap.png').
    - cmap: Color map to be used in the heatmap (default 'viridis').
    - dpi: Resolution of the saved image (default 300).
    """
    plt.figure(figsize=(8, 6))
    
    # Plot the heatmap using imshow
    heatmap = plt.imshow(data, cmap=cmap, aspect='auto')

    # Add a color bar for intensity
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Intensity', rotation=270, labelpad=15)

    # Save the heatmap to a file
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')

def create_bar_plots_in_subplots(rank_information, n_layer, filename="rank_information_plot.png"):
    """
    Creates bar plots for q-id, k-id, and v-id for each layer in a single figure using subplots.

    Parameters:
    - rank_information: Dictionary containing energy values for q-id, k-id, and v-id.
    - n_layer: Number of layers to plot.
    - filename: The name of the file to save the figure.
    """
    # Create a figure with 3 subplots per layer (one for q-id, k-id, and v-id)
    fig, axes = plt.subplots(n_layer, 3, figsize=(15, 5 * n_layer))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between plots

    for i in range(n_layer):
        # Plot for q-id
        q_values = rank_information[f"q-id{i}"]
        axes[i, 0].bar(range(len(q_values)), q_values, color='lightblue', edgecolor='black')
        axes[i, 0].set_title(f'q-id{i} Energy Values')
        axes[i, 0].set_xlabel('Energy Index')
        axes[i, 0].set_ylabel('Energy Value')
        axes[i, 0].set_xticks(range(len(q_values)))
        axes[i, 0].grid(True, axis='y', linestyle='--', alpha=0.7)

        # Plot for k-id
        k_values = rank_information[f"k-id{i}"]
        axes[i, 1].bar(range(len(k_values)), k_values, color='lightgreen', edgecolor='black')
        axes[i, 1].set_title(f'k-id{i} Energy Values')
        axes[i, 1].set_xlabel('Energy Index')
        axes[i, 1].set_ylabel('Energy Value')
        axes[i, 1].set_xticks(range(len(k_values)))
        axes[i, 1].grid(True, axis='y', linestyle='--', alpha=0.7)

        # Plot for v-id
        v_values = rank_information[f"v-id{i}"]
        axes[i, 2].bar(range(len(v_values)), v_values, color='lightcoral', edgecolor='black')
        axes[i, 2].set_title(f'v-id{i} Energy Values')
        axes[i, 2].set_xlabel('Energy Index')
        axes[i, 2].set_ylabel('Energy Value')
        axes[i, 2].set_xticks(range(len(v_values)))
        axes[i, 2].grid(True, axis='y', linestyle='--', alpha=0.7)

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()