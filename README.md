# SMDP Efficiency Reinforcement Learning Project

This repository contains the implementation of my Bachelor Project and the thesis looking into improving the efficiency of a Semi-Markov Decision Process (SMDP) based reinforcement learning algorithm. The project includes various components such as policy and value networks, managers for different SMDP variations, and utilities for running episodes and training the models.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dependencies](#dependencies)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Code Overview](#code-overview)
    - [Policy and Value Networks](#policy-and-value-networks)
    - [Managers](#managers)
    - [Workers](#workers)
    - [Wrapper](#wrapper)

## Project Structure

```
├── SMDP_single
│   ├── networks.py
│   ├── manager.py
│   ├── worker.py
├── SMDP_naive
│   ├── networks_naive.py
│   ├── manager_naive.py
│   ├── worker_naive.py
├── SMDP_head
│   ├── shared_networks.py
│   ├── manager_head.py
│   ├── worker_head.py
├── utils
│   ├── utils.py
│   ├── grid_generation.py
│   ├── step.py
│   ├── step_head.py
│   ├── prioritized_memory.py
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

## Dependencies

- Python 3.12.3
- PyTorch 2.2.1
- NumPy 2.0.0
- Gymnasium (OpenAI Gym) 0.29.1
- Matplotlib 3.5.0
- minigrid (Add wrapper) 2.2.1

## Setup and Installation

1. Clone the repository:
    ```
    git clone https://github.com/Niclas-J-M/BP.git
    cd BP
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Ensure you have the necessary PyTorch setup for your environment. Visit [PyTorch installation guide](https://pytorch.org/get-started/locally/) for detailed instructions.

## Usage

To run the SMDP algorithm, execute the `main.py` file with the appropriate configuration:

```
python main.py
```
Also set the algorithm you want to use in the python file (Multi-Headed neural network = Head, single neural network = Single, one network for each region = Naive)
Modify the `config.py` file to set different parameters such as learning rate, gamma, epsilon, etc.

## Code Overview

### Policy and Value Networks

#### DynamicPolicyNetwork and DynamicValueNetwork (in `shared_networks.py`)
These classes define dynamic policy and value networks that allow for the addition of new heads to support different options in the SMDP framework.

```python
class DynamicPolicyNetwork(nn.Module):
    # Implementation of the DynamicPolicyNetwork class

class DynamicValueNetwork(nn.Module):
    # Implementation of the DynamicValueNetwork class
```

### Managers

#### Manager_Head (in `manager_head.py`)
This class manages the policies and value networks for the SMDP head algorithm. It handles the creation of new options and updates the Q-values.

```python
class Manager_Head:
    # Implementation of the Manager_Head class
```

### Workers

#### Worker_Head (in `worker_head.py`)
This class defines the worker that interacts with the environment using dynamic heads in the policy and value networks.

```python
class Worker_Head:
    # Implementation of the Worker_Head class
```

### Wrapper
It is important to note that a wrapper has been added in the minigrid/wrapper file to run the code:

```python
class NESWActionsImage(Wrapper):
    # We change the actions to be 0:N, 1:E, 2:S, 3:W
```

This function can be found in the 'wrapper.py' file
