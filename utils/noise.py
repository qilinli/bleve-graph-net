import torch
from utils.utils import NodeType

def get_velocity_noise(graph, noise_std, noise_type, device):
    current_pressure = graph.x[:, :1]
#     timestep = graph.x[0, 1] // 0.001

#     # Time-dependent noise std
#     max_noise = noise_std
#     timesteps = 25
#     P = 2 # exponential rate, the bigger the smaller noise at early stage
#     current_noise_std = max_noise * (timestep / timesteps) ** P
    if noise_type == 'scale':
        scaling = torch.normal(std=noise_std, mean=0.0, size=current_pressure.shape).to(device)
        noise = current_pressure * scaling
    elif noise_type == 'additive':
        noise = torch.normal(std=noise_std, mean=0.0, size=current_pressure.shape).to(device)
    
    # mask = type!=NodeType.NORMAL
    # noise[mask]=0
    return noise.to(device)