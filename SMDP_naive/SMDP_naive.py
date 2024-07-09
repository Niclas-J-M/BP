# Import necessary libraries
import numpy as np
import random
from utils.utils import compression_function, print_q_table, plot_steps, plot_rewards
from utils.grid_generation import create_regions
from utils.step import run_episode, explore
from SMDP_naive.NaiveManager import NaiveManager

def SMDP_naive(env, num_regions, num_actions, num_episodes, tasks, device):
    """
    Run the SMDP naive algorithm in the given environment.

    Parameters:
    - env: The environment to run the algorithm in.
    - num_regions: Number of regions to divide the state space into.
    - num_actions: Number of possible actions in the environment.
    - num_episodes: Number of episodes to run.
    - tasks: List of tasks to accomplish.
    - device: Device to run the computations on (CPU or GPU).

    Returns:
    - total_steps_per_episode: List of total steps taken in each episode.
    - total_reward_per_episode: List of total rewards received in each episode.
    """
    size = env.unwrapped.width
    Region_bound = create_regions(size, num_regions)  # Create regions in the state space
    print(Region_bound)
    manager = NaiveManager(num_regions, num_actions, tasks, device)  # Initialize manager
    rand_seed = random.randint(1, 100)  # Random seed for environment reset
    task = 0

    total_steps_per_episode = []
    total_reward_per_episode = []

    for episodes in range(num_episodes):
        state = env.reset(seed=rand_seed)
        task = 0
        total_steps = 0
        intended_transitions = []
        total_reward = 0
        total_steps_iteration = 0

        region = compression_function(state, Region_bound)
        region_num_states = (Region_bound[region][1][1] - Region_bound[region][0][1] + 1) * (Region_bound[region][1][0] - Region_bound[region][0][0] + 1)
        print("Spawn Region", region)
        manager.add_region(region, task)

        while True:
            actual_end_region = 0

            # Task progression logic
            key = env.key
            door = env.door
            if task == 0 and key:
                task = 1
            elif task == 1 and door:
                task = 2

            region = compression_function(state, Region_bound)
            region_num_states = (Region_bound[region][1][1] - Region_bound[region][0][1] + 1) * (Region_bound[region][1][0] - Region_bound[region][0][0] + 1)
            sqrt_num = int(np.sqrt(region_num_states))
            if sqrt_num * sqrt_num != region_num_states:
                print("Error: regions not perfect square")
                return

            # Check if options exist in state space
            if manager.options_in_state_space(region, task):
                goal_region = manager.select_action(region, task)
                if goal_region > num_regions:
                    worker = manager.get_task_specific_worker(region, goal_region, region_num_states, num_actions, task)
                else:
                    worker = manager.get_create_region_option(region, region_num_states, num_actions)

                transitions, intended_transitions, reward, current_state, done, actual_end_region, steps, total_steps_iteration = run_episode(env, state, worker, goal_region, region, region_num_states, Region_bound, task, total_steps_iteration)
                total_steps_iteration += steps

                if actual_end_region != goal_region and actual_end_region != region:
                    manager.add_option(region, actual_end_region, task)
                
            else:
                transitions, reward, initial_region, goal_region, current_state, done, steps = explore(env, state, region, region_num_states, task, Region_bound)

                if initial_region not in manager.Q[task] and initial_region < num_regions + 1:
                    manager.add_region(initial_region, task)
                if goal_region not in manager.Q[task] and goal_region < num_regions + 1:
                    manager.add_region(goal_region, task)
                manager.add_option(region, goal_region, task)
                if goal_region > num_regions:
                    worker = manager.get_task_specific_worker(region, goal_region, region_num_states, num_actions, task)
                else:
                    worker = manager.get_create_region_option(region, region_num_states, num_actions)

            total_steps += steps
            total_reward += reward

            # Train intended worker
            if intended_transitions:
                worker.train(intended_transitions)
                for _ in range(4):
                    worker.train_sil()

            # Train worker
            if transitions:
                worker.train(transitions)
                for _ in range(4):
                    worker.train_sil()

            option = goal_region
            if goal_region > num_regions:
                goal_region = region
            if actual_end_region > num_regions and actual_end_region != 0:
                option = actual_end_region
                goal_region = region

            manager.update_policy(task, region, option, reward, goal_region)
            state = current_state

            if done or total_steps > 300:
                total_reward_per_episode.append(total_reward)
                break

        print("SMDP naive")
        print_q_table(manager.Q)
        print("Episode", episodes)
        print("Total steps", total_steps)

        total_steps_per_episode.append(total_steps)
        manager.decay_epsilon()

    return total_steps_per_episode, total_reward_per_episode
