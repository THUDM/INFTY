#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import math
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from infty.utils.hessian import hessian
import os
import torch



def visualize_esd(optimizer, model, create_loss_fn, loader, task, device, dir_path="./plots/esd"):
    """
    Estimate and visualize the Empirical Spectral Density (ESD) of the Hessian for a given model and task.

    Args:
        optimizer (InftyOptimizer): The Infty optimizer.
        model (torch.nn.Module): The model to analyze.
        create_loss_fn (Callable): A function that returns the loss given model, inputs, targets.
        loader (DataLoader): DataLoader for the evaluation dataset.
        task (int): Current task index for saving results.
        device (torch.device): Device to perform computations on.
        dir_path (str): Directory to store the outputs (trace and ESD data).
    """
    model.eval()
    
    print(f"{'='*30}\n[ESD] Computing Hessian trace and spectrum for task {task}...\n{'='*30}")

    # Compute trace (diagonal elements of Hessian)
    hessian_comp = hessian(model, create_loss_fn, dataloader=loader, device=device)
    
    save_path = f"{dir_path}/{optimizer.name}"
    os.makedirs(save_path, exist_ok=True)
    trace_path = f"{save_path}/trace_task{task}.pt"
    esd_path = f"{save_path}/esd_task{task}.pt"
    fig_path = f"{save_path}/fig_task{task}.pdf"

    if not os.path.exists(trace_path):
        print(f"[ESD] Estimating Hessian trace for task {task}...")
        trace = hessian_comp.trace()
        mean_trace = np.mean(trace)
        print(f"[ESD] Mean Hessian trace: {mean_trace:.4f}")
        torch.save({"mean_trace": mean_trace}, trace_path)
    else:
        trace = torch.load(trace_path, weights_only=False)
        mean_trace = trace.get("mean_trace", None)
        if mean_trace is not None:
            print(f"[ESD] Loaded mean Hessian trace: {mean_trace:.4f}")
        else:
            print(f"[ESD] Warning: 'mean_trace' not found in {trace_path}")

    # Compute empirical spectral density (ESD) of the Hessian
    if not os.path.exists(esd_path):
        print(f"[ESD] Estimating Empirical Spectral Density (ESD) for task {task}...")
        density_eigen, density_weight = hessian_comp.density()
        torch.save({"density_eigen": density_eigen, "density_weight": density_weight}, esd_path)
        print(f"[ESD] ESD data saved to '{esd_path}'.")
    else:
        print(f"[ESD] ESD file found at '{esd_path}', skipping ESD computation.")
        density_eigen = torch.load(esd_path, weights_only=False)["density_eigen"]
        density_weight = torch.load(esd_path, weights_only=False)["density_weight"] 

    print(f"[ESD] Plotting ESD for task {task}...")
    get_esd_plot(density_eigen, density_weight, fig_path)
    print(f"[ESD] Done.\n{'='*30}")



def get_esd_plot(eigenvalues, weights, fig_path):
    plt.clf()
    fig, ax = plt.subplots()

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'cm'

    fontsize = 28

    density, grids = density_generate(eigenvalues, weights)
    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=fontsize, labelpad=10)
    plt.xlabel('Eigenvalue', fontsize=fontsize, labelpad=10)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])


    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('serif') 
    ax.tick_params(axis='both', labelsize=fontsize, which='major', direction='out', length=6, width=2)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(f'{fig_path}')


def density_generate(eigenvalues,
                     weights,
                     num_bins=10000,
                     sigma_squared=1e-5,
                     overhead=0.01):
    """
    ### compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
    ### eigen: (1, 100, 2) means (number of SLQ runs, number of iterations used to compute trace, the second term is zero
    ### weights: (1, 100, 100)
    eigen, eigen_weights = torch.eig(T)

    eigenvalues = eigen
    weights = torch.pow(eigen_weights, 2)
    """

    eigenvalues = np.array(eigenvalues) # (1, 100)
    weights = np.array(weights)         # (1, 100, 100)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead  # (1, )
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead  # (1, )

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)   # (10000, 1)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))   # (1, )

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)


