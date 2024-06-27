import numpy as np
import random
from utils.utils import compression_function, print_q_table
from utils.grid_generation import create_regions
from utils.step_head import run_episode, explore
from SMDP_head.manager_head import Manager_Head

def SMDP_head(env, num_regions, num_actions, num_episodes, tasks, device):
    
    size = env.unwrapped.width
    Region_bound = create_regions(size, num_regions)
    print(Region_bound)
    manager = Manager_Head(num_regions, num_actions, tasks, device)
    rand_seed = random.randint(1, 100)
    task = 0

    total_steps_per_episode = []
    total_reward_per_episode = []
    for episodes in range(num_episodes):
        state = env.reset(seed=rand_seed)
        task = 0
        total_steps = 0
        total_reward = 0
        total_steps_iteration = 0
        
        region = compression_function(state, Region_bound)
        region_num_states =  (Region_bound[region][1][1] - Region_bound[region][0][1] + 1) * (Region_bound[region][1][0] - Region_bound[region][0][0] + 1)
        print("Spawn Region", region)
        manager.add_region(region, task)

        while True:
            actual_end_region = 0
            intended_transitions = []
            intended_worker = None
            option_intended_idx = None
            key = env.key
            door = env.door
            if task == 0 and key:
                task = 1
            elif task == 1 and door:
                task = 2

            region = compression_function(state, Region_bound)
            region_num_states =  (Region_bound[region][1][1] - Region_bound[region][0][1] + 1) * (Region_bound[region][1][0] - Region_bound[region][0][0] + 1)
            sqrt_num = int(np.sqrt(region_num_states))
            if sqrt_num * sqrt_num != region_num_states:
                print("Error: regions are not perfect squares")
                return

            if manager.options_in_state_space(region, task):
                goal_region = manager.select_action(region, task)

                if goal_region > num_regions:
                    worker = manager.get_task_specific_worker(region, goal_region, region_num_states, num_actions, task)
                else:
                    worker = manager.get_create_region_option(region, region_num_states, num_actions)
                
                if goal_region > num_regions or actual_end_region > num_regions:
                    option_idx = 0
                else:
                    option_idx = manager.option_indices[task][region][goal_region]
                

                transitions, intended_transitions, reward, current_state, done, actual_end_region, steps, total_steps_iteration = run_episode(env, state, worker, goal_region, region, region_num_states, Region_bound, task, total_steps_iteration, option_idx)
                total_steps_iteration += steps

                if actual_end_region != goal_region and actual_end_region != region:
                    intended_worker = worker
                    if actual_end_region not in manager.Q[task] and actual_end_region < num_regions + 1:
                        manager.add_region(actual_end_region, task)
                    
                    if actual_end_region not in manager.Q[task][region]:
                        if actual_end_region > num_regions:
                            worker = manager.get_task_specific_worker(region, actual_end_region, region_num_states, num_actions, task)
                        else:
                            worker = manager.get_create_region_option(region, region_num_states, num_actions)
                        manager.add_option(region, actual_end_region, task)
                        option_idx = worker.add_option_head()
                        if region < num_regions + 1 and actual_end_region < num_regions + 1:
                            for n_task in tasks:
                                manager.option_indices[n_task][region][actual_end_region] = option_idx
                        else:
                            manager.option_indices[task][region][actual_end_region] = option_idx
                    else:
                        if actual_end_region > num_regions:
                            worker = manager.get_task_specific_worker(region, actual_end_region, region_num_states, num_actions, task)
                        else:
                            worker = manager.get_create_region_option(region, region_num_states, num_actions)
                
            else:
                transitions, reward, initial_region, goal_region, current_state, done, steps = explore(env, state, region, region_num_states, task, Region_bound)

                if initial_region not in manager.Q[task] and initial_region < num_regions + 1:
                    manager.add_region(initial_region, task)
                if goal_region not in manager.Q[task] and goal_region < num_regions + 1:
                    manager.add_region(goal_region, task)

                if goal_region > num_regions + 1:
                    worker = manager.get_task_specific_worker(initial_region, goal_region, region_num_states, num_actions, task)
                else:
                    worker = manager.get_create_region_option(initial_region, region_num_states, num_actions)
                if goal_region not in manager.option_indices[task][initial_region]:
                    manager.add_option(initial_region, goal_region, task)
                    option_idx = worker.add_option_head()
                    if initial_region < num_regions + 1 and goal_region < num_regions + 1:
                        for n_task in tasks:
                            manager.option_indices[n_task][initial_region][goal_region] = option_idx
                    else:
                        manager.option_indices[task][initial_region][goal_region] = option_idx

            if goal_region > num_regions or actual_end_region > num_regions:
                option_idx = 0
                option_intended_idx = 0
            else:
                #print(goal_region, actual_end_region)
                if goal_region != actual_end_region and actual_end_region != 0 and region != actual_end_region:
                    #print("not normal")
                    #print(manager.option_indices)
                    option_idx = manager.option_indices[task][region][actual_end_region]
                    option_intended_idx = manager.option_indices[task][region][goal_region]
                else:
                    #print("normal")
                    option_idx = manager.option_indices[task][region][goal_region]

                
            total_steps += steps
            total_reward += reward
            #print("task, region, actual_end, goal_region", task, region, actual_end_region, goal_region)
            #print("intended_transitions", intended_transitions)
            if intended_transitions:
                intended_worker.train(intended_transitions, option_intended_idx)
                for _ in range(4):
                    intended_worker.train_sil(option_intended_idx)

            if transitions:
                worker.train(transitions, option_idx)
                for _ in range(4):
                    worker.train_sil(option_idx)

            option = goal_region
            if goal_region > num_regions:
                goal_region = region
            if actual_end_region > num_regions and actual_end_region != 0:
                option = actual_end_region
                goal_region = region

            manager.update_policy(task, region, option, reward, goal_region) 
            state = current_state

            if done or total_steps > 300: 
                print("total reward", total_reward)
                total_reward_per_episode.append(total_reward)
                break
        
        print("SMDP_head")
        print_q_table(manager.Q)
        print_q_table(manager.option_indices)
        print("Episodes", episodes)
        print("total steps", total_steps)
        
        total_steps_per_episode.append(total_steps)
        manager.decay_epsilon()

    return total_steps_per_episode, total_reward_per_episode
