from dataset import BLEVE_ROLLOUT
from torch_geometric.loader import DataLoader
import torch
import argparse
from tqdm import tqdm
import pickle
import torch_geometric.transforms as T
from utils.utils import NodeType
import numpy as np
from model.simulator import Simulator
from tqdm import tqdm
import os
import time

parser = argparse.ArgumentParser(description='Implementation of MeshGraphNets')

parser.add_argument("--model_dir",
                    type=str,
                    default='checkpoint/K25_ns0.05_BS4_Step300000_L10N64_long/255000-rmse0.016756052960675387.pth')

parser.add_argument("--test_split", type=str, default='test')
parser.add_argument("--rollout_num", type=int, default=5)

parser.add_argument('--graph', type=str, default='knn', help='r or knn')
parser.add_argument('--k', type=int, default=25, help='k in knn for grpah connection')
parser.add_argument('--radius', type=float, default=0.5, help='cut-off radius for graph connection')
parser.add_argument('--mode', type=str, default='rollout', help='rollout or one_step')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
parser.add_argument('--layer', type=int, default=10, help='gnn layers')
parser.add_argument('--dataset_dir', type=str, default='/home/jovyan/share/mgn_data/bleve3d/2d_Time5-55', help='dataset path')
parser.add_argument('--rollout_step', type=int, default=25, help='number of timesteps to rollout')
parser.add_argument('--output_dir', type=str, default='best', help='number of timesteps to rollout')

def rollout_error(predicteds, targets):

    number_len = targets.shape[0]
    squared_diff = np.square(predicteds - targets).reshape(number_len, -1)
    loss = np.sqrt(np.cumsum(np.mean(squared_diff, axis=1), axis=0)/np.arange(1, number_len+1))

    for show_step in range(0, number_len, 3):
        if show_step < number_len:
            print('testing rmse  @ step %d loss: %.2e'%(show_step, loss[show_step]))
        else: break

    return loss


@torch.no_grad()
def rollout(model, dataset, dataloader, transformer, mode='rollout', rollout_index=1, rollout_step=30):

    dataset.change_file(rollout_index)

    predicted_pressure = None
    mask = None
    predicteds = []
    targets = []
    
    file_index = dataset.file_index
    print(f'Rollout {file_index}...')
    
    for graph in tqdm(dataloader, total=rollout_step-1):
        graph = transformer(graph)
        graph = graph.cuda()

        if predicted_pressure is not None:
            if mode == 'rollout':
                graph.x[:, 0] = predicted_pressure.detach()         
        else:
            initial_pressure = graph.x[:, 0].detach().cpu().numpy()
        
        next_p = graph.y
        predicted_pressure = model(graph, velocity_sequence_noise=None)

        predicteds.append(predicted_pressure.detach().cpu().numpy())
        targets.append(next_p.detach().cpu().numpy())
    
    pred_p = np.concatenate((initial_pressure.reshape(1, -1), np.stack(predicteds)), axis=0)
    target_p = np.concatenate((initial_pressure.reshape(1, -1), np.stack(targets)), axis=0)
    pressure = [pred_p, target_p]

    os.makedirs('result/', exist_ok=True)
    with open(f'result/best/result_{file_index}_{dataset.graph}_{mode}.pkl', 'wb') as f:
        pickle.dump([pressure, file_index, mode], f)
    
    return pressure


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    simulator = Simulator(message_passing_num=args.layer, hidden_size=args.hidden_size, device=device, model_dir=args.model_dir)
    simulator.load_checkpoint()
    simulator.eval()

    dataset = BLEVE_ROLLOUT(args.dataset_dir, split=args.test_split, graph=args.graph, k=args.k, radius=args.radius)
    transformer = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False)])
    test_loader = DataLoader(dataset=dataset, batch_size=1)

    for i in range(args.rollout_num):
        pressure = rollout(simulator, dataset, test_loader, transformer=transformer, mode=args.mode, rollout_index=i, rollout_step=args.rollout_step)
        print('------------------------------------------------------------------')
        rollout_error(predicteds=pressure[0], targets=pressure[1])


    



