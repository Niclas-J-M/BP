from worker import Worker
import numpy as np
from config import Config

class NaiveManager:
    def __init__(self, num_states, num_actions, tasks, device):
        self.device = device
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = Config.epsilon  # Exploration rate
        self.alpha = Config.alpha  # Learning rate
        self.gamma = Config.gamma  # Discount factor
        self.tasks = tasks
        self.Q = {task: {} for task in tasks}
        self.general_workers = {}  # Single worker for general policies
        self.task_specific_workers = {}  # Separate workers for task-specific options
        self.state_space = set()

    def get_task_specific_worker(self, region_from, goal, n_states, n_actions, task):
        if (task, region_from, goal) not in self.task_specific_workers:
            self.task_specific_workers[(task, region_from, goal)] = Worker(n_states, n_actions, self.device)
        return self.task_specific_workers[(task, region_from, goal)]
    

    def get_create_region_option(self, region_from, n_states, n_actions):
        if (region_from) not in self.general_workers:
            self.general_workers[(region_from)] = Worker(n_states, n_actions, self.device)
        return self.general_workers[(region_from)]

    def update_policy(self, task, region, option, reward, next_region):
        if task in self.Q:
            if next_region in self.Q[task]:
                if not bool(self.Q[task][next_region]):
                    return
            else: return
        else: return

        max_next_q = np.max(list(self.Q[task][next_region].values())) # Get the max Q-value for the next region or 0 if empty
        td_error = reward + self.gamma * max_next_q - self.Q[task][region][option]
        self.Q[task][region][option] += self.alpha * td_error

    def select_action(self, region, task):
        # Epsilon-greedy policy for action selection    
        if np.random.random() < self.epsilon:
            return np.random.choice(list(self.Q[task][region].keys()))
        else:
            return max(self.Q[task][region], key=self.Q[task][region].get)

    def decay_epsilon(self):
        # Decay exploration rate
        self.epsilon *= Config.epsilon_decay
        self.epsilon = max(self.epsilon, Config.epsilon_min)

    def add_region(self, region, task):
        if region not in self.Q[task]:
            if region < self.num_states + 1:
                for n_task in self.tasks:
                    self.Q[n_task][region] = { } 

    def add_option(self, initital_region, goal_region, task):
        if goal_region < self.num_states + 1:
            for n_task in self.tasks:
                if initital_region != goal_region:
                    self.Q[n_task][initital_region][goal_region] = 0.0
        else:
            self.Q[task][initital_region][goal_region] = 0.0

    def options_in_state_space(self, region, task):
        if task in self.Q:
        # Check if the region is in the Q-table for the specified task
            if region in self.Q[task]:
            # Check if there are any options defined for this task and region
                return bool(self.Q[task][region])
        return False
