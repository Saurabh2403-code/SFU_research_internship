import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdyn.datasets import generate_moons
import matplotlib.animation as animation
import os
# Implement some helper functions


def eight_normal_sample(n, dim, scale=1, var=1):
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(dim), math.sqrt(var) * torch.eye(dim)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((n,))
    multi = torch.multinomial(torch.ones(8), n, replacement=True)
    data = []
    for i in range(n):
        data.append(centers[multi[i]] + noise[i])
    data = torch.stack(data)
    return data


def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1


def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.style.use('default')
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.show()



def animate(traj,file_name=animation):
    plt.style.use('dark_background')
    fig,ax=plt.subplots(figsize=(8,8),dpi=150)
    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)
    scat=ax.scatter([],[],alpha=0.8,zorder=1)
    shape=traj.shape
    def update(frame):
        positions=traj[frame]
        scat.set_offsets(positions)
        return scat
    print('Compipling Animation Frames')
    ani=animation.FuncAnimation(fig,update,frames=shape[0],interval=40,blit=False)
    output_filename = f'/Users/saurabhgiri/Desktop/SFU_Codes/conditional-flow-matching/assets/baseline-experiment/{file_name}.gif'
    print(f"Saving animation to '{output_filename}'...")
    try:
        ani.save(output_filename, writer='pillow', fps=25)
        print("Animation successfully exported!")
    except Exception as e:
        print(f"\nWarning: Could not save as GIF due to writer missing: {e}")
        print("Displaying animation plot instead...")
        plt.show()
