# Import necessary libraries
from SMDP_single.worker import Worker
import numpy as np
from config import Config

class Manager:
    def __init__(self, num_states, num_actions, tasks, device):
        """
        Initialize the Manager with parameters and configurations.

        Parameters:
        - num_states: Total number of states.
        - num_actions: Total number of actions.
        - tasks: List of tasks to manage.
        - device: Device to run the computations on (CPU or GPU).
        """
        self.device = device
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = Config.epsilon  # Exploration rate
        self.alpha = Config.alpha  # Learning rate
        self.gamma = Config.gamma  # Discount factor
        self.tasks = tasks
        self.Q = {task: {} for task in tasks}  # Q-table initialization
        self.general_workers = {}
        self.task_specific_workers = {}
        self.state_space = set()

    def get_create_region_option(self, region_from, goal, n_states, n_actions):
        """
        Get or create a general worker option for a region.

        Parameters:
        - region_from: Starting region.
        - goal: Goal region.
        - n_states: Number of states in the region.
        - n_actions: Number of actions in the region.

        Returns:
        - Worker instance for the specified region and goal.
        """
        if (region_from, goal) not in self.general_workers:
            self.general_workers[(region_from, goal)] = Worker(n_states, n_actions, self.device)
        return self.general_workers[(region_from, goal)]

    def get_create_task_option(self, region_from, goal, n_states, n_actions, task):
        """
        Get or create a task-specific worker option.

        Parameters:
        - region_from: Starting region.
        - goal: Goal region.
        - n_states: Number of states in the region.
        - n_actions: Number of actions in the region.
        - task: Specific task for the worker.

        Returns:
        - Worker instance for the specified task, region, and goal.
        """
        if (task, region_from, goal) not in self.task_specific_workers:
            self.task_specific_workers[(task, region_from, goal)] = Worker(n_states, n_actions, self.device)
        return self.task_specific_workers[(task, region_from, goal)]

    def update_policy(self, task, region, option, reward, next_region):
        """
        Update the Q-policy for a given task and region.

        Parameters:
        - task: Specific task.
        - region: Current region.
        - option: Selected option.
        - reward: Received reward.
        - next_region: Next region.
        """
        if task in self.Q and next_region in self.Q[task] and self.Q[task][next_region]:
            max_next_q = np.max(list(self.Q[task][next_region].values()))  # Max Q-value for next region
            td_error = reward + self.gamma * max_next_q - self.Q[task][region][option]
            self.Q[task][region][option] += self.alpha * td_error

    def select_action(self, region, task):
        """
        Select an action using epsilon-greedy policy.

        Parameters:
        - region: Current region.
        - task: Specific task.

        Returns:
        - Selected action (option).
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(list(self.Q[task][region].keys()))
        else:
            return max(self.Q[task][region], key=self.Q[task][region].get)

    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon) over time.
        """
        self.epsilon *= Config.epsilon_decay
        self.epsilon = max(self.epsilon, Config.epsilon_min)

    def add_region(self, region, task):
        """
        Add a region to the Q-table for all tasks.

        Parameters:
        - region: Region to add.
        - task: Specific task.
        """
        if region not in self.Q[task] and region < self.num_states + 1:
            for n_task in self.tasks:
                self.Q[n_task][region] = {}

    def add_option(self, initial_region, goal_region, task):
        """
        Add an option to the Q-table for a specific task and region.

        Parameters:
        - initial_region: Starting region.
        - goal_region: Goal region.
        - task: Specific task.
        """
        if goal_region < self.num_states + 1:
            for n_task in self.tasks:
                if initial_region != goal_region:
                    self.Q[n_task][initial_region][goal_region] = 0.0
        else:
            self.Q[task][initial_region][goal_region] = 0.0

    def options_in_state_space(self, region, task):
        """
        Check if there are options defined in the Q-table for a given task and region.

        Parameters:
        - region: Region to check.
        - task: Specific task.

        Returns:
        - Boolean indicating if options exist for the task and region.
        """
        if task in self.Q and region in self.Q[task]:
            return bool(self.Q[task][region])
        return False
