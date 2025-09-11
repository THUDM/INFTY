from tqdm import tqdm
from scipy.optimize import minimize_scalar
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
from matplotlib import cm
import os
from matplotlib import font_manager

################################################################################
#
# Define the Optimization Problem
#
################################################################################
LOWER = 0.000005

class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()
        self.centers = torch.Tensor([
            [-3.0, 0],
            [3.0, 0]])

    def forward(self, x, compute_grad=False):
        x1 = x[0]
        x2 = x[1]

        f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6
        f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2*0.5), 0)

        f1_sq = ((-x1+7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2*0.5), 0)

        f1 = f1 * c1 + f1_sq * c2
        f2 = f2 * c1 + f2_sq * c2

        f = torch.tensor([f1, f2])
        if compute_grad:
            g11 = torch.autograd.grad(f1, x1, retain_graph=True)[0].item()
            g12 = torch.autograd.grad(f1, x2, retain_graph=True)[0].item()
            g21 = torch.autograd.grad(f2, x1, retain_graph=True)[0].item()
            g22 = torch.autograd.grad(f2, x2, retain_graph=True)[0].item()
            g = torch.Tensor([[g11, g21], [g12, g22]])
            return f, g
        else:
            return f

    def batch_forward(self, x):
        x1 = x[:,0]
        x2 = x[:,1]

        f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6
        f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2*0.5), 0)

        f1_sq = ((-x1+7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        f2_sq = ((-x1-7).pow(2) + 0.1*(-x2-8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2*0.5), 0)

        f1 = f1 * c1 + f1_sq * c2
        f2 = f2 * c1 + f2_sq * c2

        f  = torch.cat([f1.view(-1, 1), f2.view(-1,1)], -1)
        return f

################################################################################
#
# Plot Utils
#
################################################################################

def plot_contour(F, init, traj, trainer, save_path="./plots/trajectory", plotbar=False):
    n = 500
    xl = 11
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    # fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    fig.subplots_adjust(left=0.01, right=0.99)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.batch_forward(Xs)

    yy = -8.3552
    Yv = Ys.mean(1)

    plt.plot(init[0], init[1], marker='o', markersize=10, zorder=20, color='k')
    plt.plot(0, yy, marker='*', markersize=15, zorder=5, color='k')
    plt.plot(7, yy, marker='None')
    for offset in [-0.2, 0, 0.2]:
        plt.gca().add_line(
            Line2D([7 - offset, 7 + offset], [yy - offset, yy + offset], color='r', linewidth=5, zorder=5))
        plt.gca().add_line(
            Line2D([7 + offset, 7 - offset], [yy - offset, yy + offset], color='r', linewidth=5, zorder=5))
    plt.plot(-7, yy, marker='None')
    for offset in [-0.2, 0, 0.2]:
        plt.gca().add_line(
            Line2D([-7 - offset, -7 + offset], [yy - offset, yy + offset], color='b', linewidth=5, zorder=5))
        plt.gca().add_line(
            Line2D([-7 + offset, -7 - offset], [yy - offset, yy + offset], color='b', linewidth=5, zorder=5))

    c = plt.contour(X, Y, Yv.view(n, n), cmap=cm.viridis, linewidths=4.0)

    sz = 36
    # l = traj.shape[0]
    # color_list = np.zeros((l, 3))
    # color_list[:, 0] = np.linspace(0.6, 0.2, l)
    # color_list[:, 1] = np.linspace(0.8, 0.3, l)
    # color_list[:, 2] = np.linspace(1.0, 0.6, l)
    if traj is not None:
        for tt in tqdm(traj):
            l = tt.shape[0]
            # color_list = np.zeros((l, 3)) # TODO: change color
            # color_list[:, 0] = 1.
            # color_list[:, 1] = np.linspace(0, 1, l)
            # color_list[:, 2] = 1 - np.linspace(0, 1, l)
            ax.scatter(tt[0], tt[1], color=cm.viridis(l), s=6, zorder=10)


    if plotbar:
        cbar = fig.colorbar(c, ticks=[-15, -10, -5, 0, 5])
        cbar.ax.tick_params(labelsize=sz)

    # ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    plt.xticks([-10, -5, 0, 5, 10], fontsize=sz, fontfamily='serif')
    plt.yticks([-10, -5, 0, 5, 10], fontsize=sz, fontfamily='serif')

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('serif') 
    ax.tick_params(axis='both', labelsize=sz, which='major', direction='out', length=6, width=2)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    plt.tight_layout()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f"{save_path}/traj_{trainer}.pdf")

    
    # scatter = ax.scatter([], [], s=5, zorder=10)

    # def update(frame):
    #     scatter.set_offsets(traj[:frame + 1])
    #     color = color_list[:frame + 1]
    #     scatter.set_facecolor(color)
    #     return scatter,

    # ani = FuncAnimation(fig, update, frames=tqdm(range(len(traj))), blit=True)
    # ani.save(f"{save_path}/animation_{optimizer_name}.mp4", writer='ffmpeg', fps=120)

    plt.close(fig)


def smooth(x, n=20):
    l = len(x)
    y = []
    for i in range(l):
        ii = max(0, i-n)
        jj = min(i+n, l-1)
        v = np.array(x[ii:jj]).astype(np.float64)
        if i < 3:
            y.append(x[i])
        else:
            y.append(v.mean())
    return y


################################################################################
#
# Optimization Solver -- Multi-Objective & Zeroth-Order Approximation
#
################################################################################

def mean_grad(x, optimizer):
    f, grads = F(x, True)
    x.grad = grads.mean(1)
    optimizer.step()
    x.grad = None
    return 

def pcgrad(x, optimizer):
    f, grads = F(x, True)
    g1 = grads[:,0]
    g2 = grads[:,1]
    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()
    if g12 < 0:
        x.grad = ((1-g12/g11)*g1+(1-g12/g22)*g2)/2
    else:
        x.grad = (g1+g2)/2
    optimizer.step()
    x.grad = None
    return

def unigrad(x, optimizer, S_T=0.1, beta=0.9):
    f, grads = F(x, True)
    g1 = grads[:,0]
    g2 = grads[:,1]

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    if g12 < S_T:
        w1 = g11 * (S_T * np.sqrt(1 - g12 ** 2) - g12 * np.sqrt(1 - S_T ** 2)) / (g22 * np.sqrt(1 - S_T ** 2) + 1e-8)
        w2 = g22 * (S_T * np.sqrt(1 - g12 ** 2) - g12 * np.sqrt(1 - S_T ** 2)) / (g11 * np.sqrt(1 - S_T ** 2) + 1e-8)
        g1_new = g1 + g2 * w1
        g2_new = g2 + g1 * w2
        S_T = (1 - beta) * S_T + beta * g12
    else:
        g1_new = g1
        g2_new = g2
    x.grad = (g1_new + g2_new) / 2
    optimizer.step()
    x.grad = None
    return

def cagrad(x, optimizer, c=0.5):
    f, grads = F(x, True)
    g1 = grads[:,0]
    g2 = grads[:,1]
    g0 = (g1+g2)/2

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11+g22+2*g12+1e-4)

    # minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = c * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-4) + \
                0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22

    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x

    gw = x * g1 + (1-x) * g2
    gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-4)

    lmbda = coef / (gw_norm+1e-4)
    g = g0 + lmbda * gw
    x.grad = g / (1+c)
    optimizer.step()
    x.grad = None
    return

@torch.no_grad()
def zo_step(x, optimizer):
    zo_random_seed = np.random.randint(10000000)
    zo_perturb_parameters(zo_random_seed, x, scaling_factor=1)
    loss1a, loss1b = zo_forward(x)
    zo_perturb_parameters(zo_random_seed, x, scaling_factor=-2)
    loss2a, loss2b = zo_forward(x)
    projected_grada = ((loss1a - loss2a) / (2 * zo_eps)).item()
    projected_gradb = ((loss1b - loss2b) / (2 * zo_eps)).item()
    zo_perturb_parameters(zo_random_seed, x, scaling_factor=1)
    torch.manual_seed(zo_random_seed)
    z = torch.normal(mean=0, std=1, size=x.data.size(), device=x.data.device, dtype=x.data.dtype)
    graddiff_times_za = projected_grada * z
    graddiff_times_zb = projected_gradb * z
    g = graddiff_times_za + graddiff_times_zb
    x.grad = g
    optimizer.step()
    x.grad = None
    return loss1a + loss1b

@torch.no_grad()
def zo_step_sign(x, optimizer):
    zo_random_seed = np.random.randint(10000000)
    zo_perturb_parameters(zo_random_seed, x, scaling_factor=1)
    loss1a, loss1b = zo_forward(x)
    zo_perturb_parameters(zo_random_seed, x, scaling_factor=-2)
    loss2a, loss2b = zo_forward(x)
    projected_grada = ((loss1a - loss2a) / (2 * zo_eps)).item()
    projected_gradb = ((loss1b - loss2b) / (2 * zo_eps)).item()
    zo_perturb_parameters(zo_random_seed, x, scaling_factor=1)
    torch.manual_seed(zo_random_seed)
    z = torch.normal(mean=0, std=1, size=x.data.size(), device=x.data.device, dtype=x.data.dtype)

    graddiff_times_za = np.sign(projected_grada) * z
    graddiff_times_zb = np.sign(projected_gradb) * z
    g = graddiff_times_za + graddiff_times_zb
    x.grad = g
    optimizer.step()
    x.grad = None
    return loss1a + loss1b

@torch.no_grad()
def zo_step_q4(x, optimizer):
    for i_q in range(4):
        zo_random_seed = np.random.randint(10000000)
        zo_perturb_parameters(zo_random_seed, x, scaling_factor=1)
        loss1a, loss1b = zo_forward(x)
        zo_perturb_parameters(zo_random_seed, x, scaling_factor=-2)
        loss2a, loss2b = zo_forward(x)
        projected_grada = ((loss1a - loss2a) / (2 * zo_eps)).item()
        projected_gradb = ((loss1b - loss2b) / (2 * zo_eps)).item()
        zo_perturb_parameters(zo_random_seed, x, scaling_factor=1)
        torch.manual_seed(zo_random_seed)
        z = torch.normal(mean=0, std=1, size=x.data.size(), device=x.data.device,
                         dtype=x.data.dtype)
        graddiff_times_za = projected_grada * z
        graddiff_times_zb = projected_gradb * z
        g = graddiff_times_za + graddiff_times_zb
        if i_q == 0:
            x.grad = g / 4
        else:
            x.grad += g / 4
    optimizer.step()
    x.grad = None
    optimizer.zero_grad()
    return loss1a + loss1b

@torch.no_grad()
def zo_conserv_step(x, optimizer):
    loss0a, loss0b = zo_forward(x)
    zo_random_seed = np.random.randint(10000000)
    zo_perturb_parameters(zo_random_seed, x, scaling_factor=1)
    loss1a, loss1b = zo_forward(x)
    zo_perturb_parameters(zo_random_seed, x, scaling_factor=-2)
    loss2a, loss2b = zo_forward(x)
    projected_grada = ((loss1a - loss2a) / (2 * zo_eps)).item()
    projected_gradb = ((loss1b - loss2b) / (2 * zo_eps)).item()
    zo_perturb_parameters(zo_random_seed, x, scaling_factor=1)
    def update_params(sign, c):
        torch.manual_seed(zo_random_seed)
        z = torch.normal(mean=0, std=1, size=x.data.size(), device=x.data.device,
                         dtype=x.data.dtype)
        graddiff_times_za = projected_grada * z
        graddiff_times_zb = projected_gradb * z
        g = sign * (graddiff_times_za + graddiff_times_zb)
        x.grad = g
        optimizer.step()
        x.grad = None
    update_params(sign=1.0, c=1)
    loss1a, loss1b = zo_forward(x)
    update_params(sign=-2.0, c=2)
    loss2a, loss2b = zo_forward(x)
    if loss1a + loss1b > loss0a + loss0b:
        if loss0a + loss0b < loss2a + loss2b:
            update_params(sign=1.0, c=3)
    else:
        if loss1a + loss1b < loss2a + loss2b:
            update_params(sign=2.0, c=3)
    return loss1a + loss1b

def zo_perturb_parameters(zo_random_seed, x, random_seed=None, scaling_factor=1):
        torch.manual_seed(random_seed if random_seed is not None else zo_random_seed)
        z = torch.normal(mean=0, std=1, size=x.data.size(), device=x.data.device, dtype=x.data.dtype)
        x.data = x.data + scaling_factor * z * zo_eps

def zo_forward(x):
    with torch.inference_mode():
        f = F(x)
        loss = f[0] + f[1]
    return f[0].detach(), f[1].detach()


### Define the problem ###
F = Toy()
zo_eps = 0.001

maps = {
    "sgd": mean_grad,
    "adam": mean_grad,
    "adamw": mean_grad,
    "pcgrad": pcgrad,
    "cagrad": cagrad,
    "unigrad": unigrad,
    "zo_adam": zo_step,
    "zo_adam_q4": zo_step_q4,
    "zo_adam_sign": zo_step_sign,
    "zo_adam_cons": zo_conserv_step,
}



def run(optimizer_name, lr, init,  n_iter):
    traj = []
    x = torch.tensor(init)
    x.requires_grad = True
    if optimizer_name == "sgd":
        opt = torch.optim.SGD([x], lr=lr)
    elif optimizer_name == "adam":
        opt = torch.optim.Adam([x], lr=lr)
    elif optimizer_name == "adamw":
        opt = torch.optim.AdamW([x], lr=lr)
    else:
        opt = torch.optim.Adam([x], lr=lr)
    for it in tqdm(range(n_iter)):
        traj.append(x.detach().numpy().copy())
        maps[optimizer_name](x, opt)
    traj = torch.tensor(np.array(traj))
    return traj

def visualize_trajectory(optimizer_name):
    init = [-8.5, 5.]
    # init = [-3.5, 5.]
    n_iter = 10000
    lr = 0.1

    # optimizer_name = ["sgd", "adam", "adamw", "pcgrad", "cagrad", "unigrad", "zo_adam", "zo_adam_q4", "zo_adam_sign", "zo_adam_cons"]
    assert optimizer_name in maps.keys()
    plot_contour(F, init, traj, optimizer_name)

if __name__ == "__main__":
    method = ["sgd", "adam", "adamw", "pcgrad", "cagrad", "unigrad", "zo_adam", "zo_adam_q4", "zo_adam_sign", "zo_adam_cons"]
    for m in method:
        print(f"Running: {m}")
        visualize_trajectory(m)
