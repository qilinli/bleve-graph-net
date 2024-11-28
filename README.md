# BLEVE Graph Network (BGN)

This repository contains PyTorch implementations of BGN from the following paper:

    Li, Qilin, Zitong Wang, Ling Li, Hong Hao, Wensu Chen, and Yanda Shao. 
    "Machine learning prediction of structural dynamic responses using graph neural networks." 
    Computers & Structures 289 (2023): 107188.

The code is significantly based on [this reporsitory](https://github.com/echowve/meshGraphNets_pytorch).

## Setup

- The shell script 'create_conda_env.sh' provides a way of initialising conda environment.
- A snapshot of the complete conda environment is also given in 'environment.yml'.

## Sample usage

- Download the BLEVE 2D dataset from [OneDrive](https://curtin-my.sharepoint.com/:f:/g/personal/272766h_curtin_edu_au/ElTSySAZfjNBiaTP1bXKvUYBXBe8qM3kj6wGqqSDDw_t0w)

- Training a BGN model with 
  - A k-nearst neighbour graph (k=25), 10 message-passing steps (layers), with 64 hidden neurons in all MLP layers. 
  - The training is scheduled for a maximum of 300K iterations with an initial learning rate of 1e-3 and batch size of 4. 
  - Gaussian additive noise is added with std of 0.05 to mitigate error accumulation in the rollout inference. 
  - The code also does evaluation on the validation set every 5K iterations. 
  - For a full list of adjustable hyperparameters, please see [train_bleve.py].
    ```
    train_bleve.py --graph=knn \
                   --k=25 \
                   --hidden_size=64 \
                   --layer=10 \
                   --lr_init=1e-3 \
                   --max_step=300000 \
                   --dataset_dir=data/bleve3d/2d_Time5-55/ \
                   --batch_size=4 \
                   --eval_step=5000 \
                   --noise_std=0.05 \
                   --noise_type=additive \
                   --print_step=10 \
                   --rollout_step=25 \
                   --log
    ```
 ## Demo
 
These are the results from BGN (prediction) and FLACS CFD simulation (target). The animation shows the blast pressure wave with the unit in bar.

<img src="https://github.com/qilinli/bleve-graph-net/blob/feb6f602ff05626d4c2b460f9bd5f6b58346441b/images/100046_knn_rollout.gif?raw=true" />

<img src="https://github.com/qilinli/bleve-graph-net/blob/main/images/200130_knn_rollout.gif?raw=true" />