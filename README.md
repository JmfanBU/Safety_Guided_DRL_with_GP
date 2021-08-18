# Safety_Guided_DRL_with_GP
This repository is the official implementation of Paper ["Safety-guided deep reinforcement learning via online gaussian process estimation"](https://arxiv.org/pdf/1903.02526.pdf).

## Prerequisites
Our implementation is based on <a href="https://github.com/GPflow/GPflow.git/">GPflow</a> and <a href="https://github.com/openai/baselines">OpenAI Baselines</a>.

### Tensorflow and GPflow
In our implementation, we use Tensorflow 1.12.0 and GPflow 1.4.1.
- You can install Tensorflow via
    ```bash
    pip install tensorflow-gpu==1.12.0  # if you have a CUDA-compatible gpu and proper drivers
    ```
    or
    ```bash
    pip install tensorflow==1.12.0
    ```

- Install GPflow via
    ```bash
    pip install gpflow==1.4.1
    ```

### OpenAI Baseliens
You can find detailed instructions for installing OpenAI Baselines <a href="https://github.com/openai/baselines">here</a>.

Our implementation is based on a commit [c57528573ea695b19cd03e98dae48f0082fb2b5e](https://github.com/openai/baselines/tree/c57528573ea695b19cd03e98dae48f0082fb2b5e)

### MuJoCO
Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

The MuJoCo environments used in our paper depend on <a href="https://github.com/openai/gym">OpenAI Gym</a> as well.

## Installation

Run the following command from the project directory:

```bash
pip install -e .
```


## How to use

Our implementation includes two methods: vanilla DDPG and DDPG with
online GP estimation.

To train a vanilla ddpg policy, use the code in [ddpg_baseline](SafetyGuided_DRL/ddpg_baseline).

For DDPG using online GP, use the code in [safe_ddpg](SafetyGuided_DRL/safe_ddpg).

As default, training results will be saved to [data_ddpg](SafetyGuided_DRL/data_ddpgn).

We also provide some samples of console outputs in [outputs](SafetyGuided_DRL/outputs).

### DDPG_with_Online_GP_Estimation

An example of training DDPG with online GP policy for pendulum can be found in [train](train):

```bash
./train/pendulum_0.1M_safe_ddpg.sh
```

### Vanilla DDPG
To train with vanila DDPG:

```bash
./train/pendulum_1M_ddpg.sh
```

### DDPG_with_init_GP

```bash
./train/half_cheetah_0.1M_init_safe_ddpg.sh
```
