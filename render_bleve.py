import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import glob
import argparse
import os
import os.path as osp

parser = argparse.ArgumentParser(description='Implementation of MeshGraphNets')
parser.add_argument("--indir", type=str, default='/home/jovyan/work/mgn_2D/result/', help="directory contains rollout output")
parser.add_argument("--rollout_num", type=int, default=3)
parser.add_argument("--outdir", type=str, default='videos/')
parser.add_argument("--graph", type=str, default='knn', help='r or knn')
parser.add_argument("--mode", type=str, default='rollout', help='rollout or one_step')

args = parser.parse_args()

cb_cap = 0.1
vis_cap = 0.01
s = 5

grid = np.load('/home/jovyan/work/mgn_2D/data/bleve3d/2d_Time5-35/grid.npy')
x = grid[..., 0].flatten()
y = grid[..., 1].flatten()

rollout_files = glob.glob(args.indir + f'*100018_{args.graph}_{args.mode}.pkl')

for idx in range(args.rollout_num):
    rollout_file = rollout_files[idx]
    with open(rollout_file, "rb") as file:
        result = pickle.load(file)
        
    file_index = result[1]
    outpath = osp.join(args.outdir, file_index)
    os.makedirs(outpath, exist_ok=True)
    
    fig = plt.figure(figsize=plt.figaspect(0.3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    axes = [ax1, ax2]
    
    print('Render {} in {}...'.format(file_index, rollout_file.split('/')[-1]))
    plot_info = []
    for idx, label in enumerate(['prediction', 'target']):
        pressure = result[0][idx]
        print(label, 'mean std min max at step20', pressure[20].mean(), pressure[20].std(), pressure[20].min(), pressure[20].max())
        ax = axes[idx]
        ax.set_xlim((-20, 20))
        ax.set_ylim((-20, 20))

        scatter = ax.scatter(x, 
                             y, 
                             c=pressure[-1],
                             cmap='bwr',
                             vmin=-cb_cap,
                             vmax=cb_cap,
                             s=s, 
                             edgecolors='none')
        cbar = fig.colorbar(scatter)
        plot_info.append((ax, label, scatter, pressure))


    def anim(frame):
        frames_to_save = [0, 3, 5, 7, 10, 15, 20, 30, 35, 40, 45, 50]

        # Update the pressure data for each timestep
        for ax, label, scatter, pressure in plot_info:
            p = pressure[frame]
            mask = np.logical_and(p > -vis_cap, p < vis_cap)
            p[mask] = np.nan

            scatter.set_array(p.flatten())
            ax.set_title('{}\nTime @ {} ms'.format(label, frame))
            ax.grid(True, linestyle='--')
        if frame in frames_to_save:
            frame_name = '{}_frame{:03d}_{}_{}.png'.format(file_index, frame, args.graph, args.mode)
            frame_path = osp.join(outpath, frame_name)
            plt.savefig(frame_path, dpi=100)
        return scatter

    unused_animation = animation.FuncAnimation(
        fig, 
        anim,
        frames=np.arange(0, pressure.shape[0], 1), 
        interval=10)

    unused_animation.save(f'{outpath}/{file_index}_{args.graph}_{args.mode}.gif', dpi=100, fps=5, writer='Pillow')

