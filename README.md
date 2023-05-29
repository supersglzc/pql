# Parallel Q-Learning (PQL)
This repository provides an implementation of the paper "Parallel *Q*-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation".

- [Installation](#installation)
    - [Install :zap: PQL](#install_pql)
    - [Install Isaac Gym](#install_isaac)
- [System Requirements](#requirements)
- [Usage](#usage)
    - [Train with :zap: PQL](#usage_pql)
    - [Baselines](#usage_baselines)
    - [Logging](#usage_logging)
    - [Saving and Loading](#usage_saving_loading)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation

### Install :zap: PQL <a name="install_pql"></a>

1. Clone the package:

    ```bash
    git clone git@github.com:Improbable-AI/pql.git
    cd pql
    ```

2. Create Conda environment and install dependencies:

    ```bash
    ./create_conda_env_pql.sh
    pip install -e .
    ```


### Install Isaac Gym <a name="install_isaac"></a>

> **Note**
> In original paper, we use Isaac Gym Preview 3 and task configs in commit ca7a4fb762f9581e39cc2aab644f18a83d6ab0ba in IsaacGymEnvs.

1. Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym

2. Unzip the file:
    ```bash
    tar -xf IsaacGym_Preview_4_Package.tar.gz
    ```

3. Install IsaacGym
    ```bash
    cd isaacgym/python
    pip install -e . --no-deps
    ```

5. Install IsaacGymEnvs

    ```bash
    git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
    cd IsaacGymEnvs
    pip install -e . --no-deps
    ```
    
6. Export LIBRARY_PATH
    
    ```bash
    export LD_LIBRARY_PATH=$(conda info --base)/envs/pql/lib/:$LD_LIBRARY_PATH
    ```

## System Requirements <a name="requirements"></a>
> **Warning**
> Note that wall-clock efficiency highly depends on the GPU type and will decrease with smaller/fewer GPUs (check Section 4.4 in the paper).

Isaac Gym requires an NVIDIA GPU. To train in the default configuration, we recommend a GPU with at least 10GB of VRAM. For smaller GPUs, you can decrease the number of parallel environments (`cfg.num_envs`), batch_size (`cfg.algo.batch_size`), replay buffer capacity (`cfg.algo.memory_size`), etc. :zap: PQL can run on 1/2/3 GPUs (set GPU ID `cfg.p_learner_gpu` and `cfg.v_learner_gpu`; default GPU ID for Isaac Gym env is `GPU:0`). 


## Usage

### Train with :zap: PQL <a name="usage_pql"></a>

Run :zap: PQL on Shadow Hand task. A full list of tasks in Isaac Gym is available [here](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/rl_examples.md).

```bash
python train_pql.py task=ShadowHand
```

Run :zap: PQL-D (with distributional RL)

```bash
python train_pql.py task=ShadowHand algo.distl=True algo.cri_class=DistributionalDoubleQ
```

### Baselines <a name="usage_baselines"></a>

Run DDPG baseline

```bash
python train_baselines.py algo=ddpg_algo task=ShadowHand
```

Run SAC baseline

```bash
python train_baselines.py algo=sac_algo task=ShadowHand
```

Run PPO baseline

```bash
python train_baselines.py algo=ppo_algo task=ShadowHand isaac_param=True
```

### Logging <a name="usage_logging"></a>

We use Weights & Biases (W&B) for logging. 

1. Get a W&B account from https://wandb.ai/site

2. Get your API key from https://wandb.ai/authorize

3. set up your account in terminal
    ```bash
    export WANDB_API_KEY=$API Key$
    ```
    

### Saving and Loading <a name="usage_saving_loading"></a>

Checkpoints are automatically saved as W&B [Artifacts](https://docs.wandb.ai/ref/python/artifact).

To load and visualize the policy, run

```bash
python visualize.py task=ShadowHand headless=False num_envs=10 artifact=$team-name$/$project-name$/$run-id$/$version$
```


## Citation

```
@inproceedings{li2023pql,
  title={Parallel $Q$-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation},
  author={Li, Zechu and Chen, Tao and Hong, Zhang-Wei and Ajay, Anurag and Agrawal, Pulkit},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```

## Acknowledgement

We thank the members of the Improbable AI lab for the helpful discussions and feedback on the paper. We are grateful to MIT Supercloud and the Lincoln Laboratory Supercomputing Center for providing HPC resources.
