import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def visualize_conflicts(optimizer, task_id, dir_path="./plots/conflicts"):
    """
    Visualize conflicts and plot cosine similarity during training process
    Args:
        optimizer: optimizer object
        task_id: task identity
        dir_path: the path to save the plot
    """
    assert optimizer.name in ["pcgrad", "cagrad", "unigrad_fs", "gradvac"]
    if task_id > 0:
        sim_list = optimizer.sim_list

        save_path = f"{dir_path}/{optimizer.name}"
        os.makedirs(save_path, exist_ok=True)
        sim_path = f"{save_path}/sim_list_task{task_id}.pt"
        torch.save(sim_list, sim_path)
        print(f"Cosine similarity saved to: {sim_path}")

        print(f"Cosine similarity range: [{min(sim_list):.4f}, {max(sim_list):.4f}]")      
        print(f"Mean: {np.mean(sim_list):.4f}, Std: {np.std(sim_list):.4f}")
        
        iterations = list(range(1, len(sim_list) + 1))
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(iterations, sim_list, 'b-', linewidth=2, alpha=0.8, label='Cosine Similarity')
        plt.grid(True, alpha=0.3)
        
        fontsize = 24

        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel('Iterations', fontsize=fontsize)
        plt.ylabel('Cosine Similarity', fontsize=fontsize)
        
        plt.xlim(1, len(sim_list))
        plt.ylim(min(sim_list) - 0.05, max(sim_list) + 0.05)
        
        mean_sim = np.mean(sim_list)
        std_sim = np.std(sim_list)
        plt.text(0.02, 0.98, f'Mean: {mean_sim:.4f}\nStd: {std_sim:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top', fontsize=fontsize,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        plt.savefig(f"{dir_path}/similarity_{task_id}.pdf", dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {dir_path}/similarity_{task_id}.pdf")
    
