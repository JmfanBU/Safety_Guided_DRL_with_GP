# Safety_Guided_DRL_with_GP
This repository is the official implementation of Paper ["Safety-guided deep reinforcement learning via online gaussian process estimation"](https://arxiv.org/pdf/1903.02526.pdf).

## Prerequisites
Our implementation is based on  <a href="https://github.com/openai/baselines">OpenAI Baselines</a> and <a href="https://github.com/GPflow/GPflow.git/">GPflow</a>.

### OpenAI Baseliens
You can find detailed instructions for installing OpenAI Baselines <a href="https://github.com/openai/baselines">here</a>.

Our implementation is based on a commit [c57528573ea695b19cd03e98dae48f0082fb2b5e](https://github.com/openai/baselines/tree/c57528573ea695b19cd03e98dae48f0082fb2b5e)

### GPflow
GPflow can be installed with:
```bash
pip install gpflow==1.4.1
```

### MuJoCO
Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

The MuJoCo environments used in our paper depend on <a href="https://github.com/openai/gym">OpenAI Gym</a> as well.

## Installation

Run the following command from the project directory:

```bash
pip install -e .
```


## How to use

### DDPG_with_GP

DDPG with GP has two components: vanila ddpg and online gp approximation.

To train a vanila ddpg policy, use the code in [ddpg_baseline](SafetyGuided_DRL/ddpg_baseline)

For DDPG using GP, use the code in [safe_ddpg](SafetyGuided_DRL/safe_ddpg).

An example of DDPG with GP policy for pendulum can be found in [train](train):

```bash
./train/pendulum_0.1M_safe_ddpg.sh
```

This is for vanila ddpg:

```bash
./train/pendulum_1M_ddpg.sh
```

The training results will be saved to data_ddpg/.

### DDPG_with_init_GP

```bash
./train/half_cheetah_0.1M_init_safe_ddpg.sh
```
