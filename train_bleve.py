from dataset import BLEVE, BLEVE_ROLLOUT
from model.simulator import Simulator
import torch
from utils.noise import get_velocity_noise
import time
from utils.utils import NodeType, update_graph
from rollout_bleve import rollout, rollout_error
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import os
import wandb
import argparse
import numpy as np
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# Add arguments to the parser
# Data
parser.add_argument('--dataset_dir', type=str, default='data/bleve/2d_Time5-55', help='dataset path')

# Graph connection
parser.add_argument('--graph', type=str, default='knn', help='r or knn')
parser.add_argument('--k', type=int, default=12, help='k in knn for grpah connection')
parser.add_argument('--radius', type=float, default=0.5, help='cut-off radius for graph connection')

# GNN architecture
parser.add_argument('--layer', type=int, default=5, help='number of gnn layers')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden layer dimension in gnn')

# Learning rate schedule
parser.add_argument('--lr_init', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--lr_decay', type=float, default=0.1, help='lr decay rate')
parser.add_argument('--lr_decay_step', type=int, default=3e4, help='lr decay step')
parser.add_argument('--max_step', type=int, default=150000, help='max number of training iterations')

# Training and optimization
parser.add_argument('--noise_type', type=str, default='additive', help='additive or scale Gaussian noise')
parser.add_argument('--noise_std', type=float, default=0, help='scale of the noise to pressure')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--ar_step', type=int, default=1e7, help='auto regressive starting step, 1e7 means no ar')
parser.add_argument('--ar2_step', type=int, default=1e7, help='2nd auto regressive starting step, 1e7 means no ar')
parser.add_argument('--rollout_step', type=int, default=30, help='number of timestep to rollout')

# Logging
parser.add_argument('--log', action='store_true', help='Enable wandb log')
parser.add_argument('--eval_step', type=int, default=1e4, help='validation every eval_step iteration')
parser.add_argument('--print_step', type=int, default=10, help='print loss every print_step iteration')
parser.add_argument('--run_name_postfix', type=str, default='', help='dataset path')


def train(dataloader, args):
    
    simulator = Simulator(message_passing_num=args.layer, hidden_size=args.hidden_size, device=device)
    optimizer= torch.optim.AdamW(simulator.parameters(), lr=args.lr_init)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=100000,
                                              cycle_mult=1,
                                              max_lr=0.001,
                                              min_lr=0,
                                              warmup_steps=1000,
                                              gamma=0.1)
    
    best_eval_loss = 100
    for step, graph in enumerate(dataloader):
        loss = 0
        log = {}  # wandb log
        simulator.train()
        graph = transformer(graph)
        graph = graph.cuda()
        frame_time = graph.x[0, 1] # elapsed time of the input frame
        # print('Graph before normalisation')
        #idx = [x for x in range(4870, 4871)]
        # print('graph.x', graph.x[idx, :])
        # print('graph.y', graph.y[idx])
        pressure_sequence_noise = 0
        if args.noise_std > 0:
            pressure_sequence_noise = get_velocity_noise(graph, noise_std=args.noise_std, noise_type=args.noise_type, device=device)
        predicted_p, target_p = simulator(graph, pressure_sequence_noise)
        # print('Predicted_p', predicted_p[idx])
        # print('Graph after normalisation')
        # print('graph.x', graph.x[idx, :])
        #print('graph.y', graph.y[idx], graph.next_y[idx], graph.next2_y[idx])
        
        errors = ((predicted_p - target_p)**2)
        loss1 = torch.mean(errors)
        
        if step > args.ar_step:
            graph2 = update_graph(graph, predicted_p)
            # print('Graph2 before normalisation')
            # print('graph2.x', graph2.x[idx, :])
            # print('graph2.y', graph2.y[idx])
            node_attr = simulator._node_normalizer.inverse(graph2.x)
            # print('Graph2 node attr after inverse norm')
            # print(node_attr[idx, :])
            node_attr[:, -1] += 0.001
            graph2.x = simulator._node_normalizer(node_attr, accumulate=True)
            # print('Graph2 after normalisation')
            # print('graph2.x', graph2.x[idx, :])
            #print('graph2.y', graph2.y[idx], graph2.next_y[idx], graph2.next2_y[idx])
            predicted_p2 = simulator.model(graph2).squeeze()
            target_p2 = simulator._output_normalizer(graph2.y, accumulate=True).squeeze()
            # print('Predicted_p2', predicted_p2[idx])
            errors2 = ((predicted_p2 - target_p2)**2)
            loss2 = torch.mean(errors2)
        else:
            loss2 = torch.zeros((1)).cuda()
        
        if step > args.ar2_step:
            graph3 = update_graph(graph2, predicted_p2)
            # print('Graph2 before normalisation')
            # print('graph2.x', graph2.x[idx, :])
            # print('graph2.y', graph2.y[idx])
            node_attr = simulator._node_normalizer.inverse(graph3.x)
            # print('Graph2 node attr after inverse norm')
            # print(node_attr[idx, :])
            node_attr[:, -1] += 0.001
            graph3.x = simulator._node_normalizer(node_attr, accumulate=True)
            # print('Graph2 after normalisation')
            # print('graph2.x', graph2.x[idx, :])
            #print('graph3.y', graph3.y[idx], graph3.next_y[idx], graph3.next2_y[idx])
            predicted_p3 = simulator.model(graph3).squeeze()
            target_p3 = simulator._output_normalizer(graph3.y, accumulate=True).squeeze()
            # print('Predicted_p2', predicted_p2[idx])
            errors2 = ((predicted_p3 - target_p3)**2)
            loss3 = torch.mean(errors2)
        else:
            loss3 = torch.zeros((1)).cuda()
        
        loss = loss1 + loss2 + loss3
        
        if not torch.isnan(loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
              
        # Log
        if step % args.print_step == 0:
            print('Step %d [loss %.2e], [loss1 %.2e], [loss2 %.2e], [loss3 %.2e], [lr %.2e], [time %d]' % (
                step, loss.item(), loss1.item(), loss2.item(), loss3.item(), scheduler.get_lr()[0], frame_time.item()*1000))
            log['train/lr'] = scheduler.get_lr()[0]
            log['train/loss'] = loss.item()
            log['train/loss1'] = loss1.item()
            log['train/loss2'] = loss2.item()
            log['train/loss3'] = loss3.item()
            
        # Evaluation
        if step != 0 and step % args.eval_step == 0:
            with torch.no_grad():
                simulator.eval()
                dataset = BLEVE_ROLLOUT(args.dataset_dir, split='valid', graph=args.graph, k=args.k, radius=args.radius)
                test_loader = DataLoader(dataset=dataset, batch_size=1)

                eval_losses = []
                for i in range(10):
                    pressure = rollout(simulator, dataset, test_loader, transformer=transformer, rollout_index=i, rollout_step=args.rollout_step)
                    print('------------------------------------------------------------------')
                    eval_loss = rollout_error(predicteds=pressure[0], targets=pressure[1])  # this is accumulated rmse
                    eval_losses.append(eval_loss)
                eval_losses = np.vstack(eval_losses).transpose(1, 0) # shape [timestep, bleve_cases]
                log['val/loss_1_step'] = eval_losses[1, :].mean()
                log['val/loss_25_step'] = eval_losses[args.rollout_step // 2, :].mean()
                log['val/loss_50_step'] = eval_losses[-1, :].mean()
                
            # Save the model
            if eval_losses[-1, :].mean() < best_eval_loss:
                best_eval_loss = eval_losses[-1, :].mean()
                if args.graph == 'r':
                    ckpt_dir = f'checkpoint/K{args.k}_ns{args.noise_std}_BS{args.batch_size}_Step{args.max_step}_L{args.layer}N{args.hidden_size}_{args.run_name_postfix}'
                else:
                    ckpt_dir = f'checkpoint/K{args.k}_ns{args.noise_std}_BS{args.batch_size}_Step{args.max_step}_L{args.layer}N{args.hidden_size}_{args.run_name_postfix}'
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                simulator.save_checkpoint(savedir=f'{ckpt_dir}/{step}-rmse{best_eval_loss}.pth')
                np.savez(f'{ckpt_dir}/{step}-rmse{best_eval_loss}', pred=predicted_p.detach().cpu().numpy(), target=target_p.detach().cpu().numpy())
        
        if args.log:
            wandb.log(log, step=step)
        if step >= args.max_step:
            break
                

if __name__ == '__main__':
    args = parser.parse_args()
    print("Command line arguments:")
    model_dir = f'checkpoint/K{args.k}_ns{args.noise_std}_BS{args.batch_size}_Step{args.max_step}_L{args.layer}N{args.hidden_size}_{args.run_name_postfix}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    with open(model_dir + '/args.txt', 'a') as f:      
        for arg in vars(args):
            print(f"    {arg}: {getattr(args, arg)}")
            f.write(f'{arg}: {getattr(args, arg)}\n')
            
    dataset = BLEVE(dataset_dir=args.dataset_dir, split='train', graph=args.graph, k=args.k, radius=args.radius)
    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=4)
    transformer = T.Compose([T.Cartesian(norm=False), T.Distance(norm=False)])

    if args.log:
        if args.graph == 'r':
            run_name = f'R{args.k}_ns{args.noise_std}_BS{args.batch_size}_Step{args.max_step}_L{args.layer}N{args.hidden_size}_' + args.run_name_postfix
        else:
            run_name = f'K{args.k}_ns{args.noise_std}_BS{args.batch_size}_Step{args.max_step}_L{args.layer}N{args.hidden_size}_' + args.run_name_postfix
            
        wandb.init(project='BLEVE-2D-50Steps', name=run_name)
        train(train_loader, args)
        wandb.finish()
    else:
        train(train_loader, args)