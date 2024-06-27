from utils.utils import coord_to_state, compression_function
import random


"""
actions to be 0:N, 1:E, 2:S, 3:W
"""


def run_episode(env, state, worker, end_region, initial_region, num_states, Region_bound, task, total_steps_iteration, step_limit=6):
    state_vector = coord_to_state(state, num_states, Region_bound)
    done = False
    final_done = False
    transitions = []
    total_reward = 0
    steps = 0
    current_state = state
    actual_end_region = initial_region

    while not done and steps < step_limit:
        if random.random() < 0.20:
            # Choose a random action with 20% probability
            action = random.randint(0, 3)
        else:
            # Choose the action suggested by the worker the rest of the time (80%)
            action = worker.select_action(state_vector)
        next_state_coord, reward, done, _ = env.step(action)
        #print(reward)
        steps += 1
        key = env.key
        door = env.door
        if task == 0:
            if key:
                total_reward = 1
                done = True
                next_state_vector = coord_to_state(next_state_coord, num_states, Region_bound)
                transitions.append((state_vector, action, total_reward, next_state_vector, done))
                actual_end_region = 101
                return transitions, total_reward, next_state_coord, final_done, actual_end_region, steps
        elif task == 1:
            if door:
                total_reward = 1
                done = True
                next_state_vector = coord_to_state(next_state_coord, num_states, Region_bound)
                transitions.append((state_vector, action, total_reward, next_state_vector, done))
                actual_end_region = 102
                return transitions, total_reward, next_state_coord, final_done, actual_end_region, steps
        elif task == 2:
            if done:
                final_done = True
                total_reward = 1
                done = True
                next_state_vector = coord_to_state(next_state_coord, num_states, Region_bound)
                transitions.append((state_vector, action, total_reward, next_state_vector, done))
                actual_end_region = 103
                return transitions, total_reward, next_state_coord, final_done, actual_end_region, steps


        current_region = compression_function(next_state_coord, Region_bound)
        #print("current_region", current_region)

        # Termination of option
        if current_region != initial_region:
            current_state = next_state_coord
            done = True
            if current_region == end_region:
                #print("well done", end_region)
                actual_end_region = end_region
                reward = 0.8
            else:
                reward = 0.8
                actual_end_region = current_region
            next_state_vector = state_vector # set the next position to be the same position
            transitions.append((state_vector, action, reward, next_state_vector, done))
            #print("actual", actual_end_region)
            return transitions, total_reward, current_state, final_done, actual_end_region, steps
            

        next_state_vector = coord_to_state(next_state_coord, num_states, Region_bound)

        transitions.append((state_vector, action, reward, next_state_vector, done))
        state_vector = next_state_vector

    if not done and total_steps_iteration >= 100:
        total_reward -= 0.1

    return transitions, total_reward, current_state, final_done, actual_end_region, steps


def explore(env, state, initial_region, num_states, task, Region_bound, step_limit=6):
    current_state_vector = coord_to_state(state, num_states, Region_bound)
    transitions = []
    done = False
    total_reward = 0
    final_done = False
    steps = 0
    while not done and steps < step_limit:
        action = random.randint(0, 3)
        next_state_coord, reward, done, _ = env.step(action)
        steps += 1
        next_state_vector = coord_to_state(next_state_coord, num_states, Region_bound)
        next_region = compression_function(next_state_coord, Region_bound)
        #print("iniitial_region", initial_region)
        #print("next_region", next_region)
        key = env.key
        door = env.door
        if done:
            total_reward = 1
            final_done = True
            next_region = 103
        if task == 0 and key:
            total_reward = 1
            done = True
            next_region = 101
        if task == 1 and door:
            total_reward = 1
            next_region = 102
            done = True

        if initial_region != next_region:
            next_state_vector = current_state_vector
            reward = 0.8
            done = True
        if total_reward >= 0.9:
            reward = total_reward
        transitions.append((current_state_vector, action, reward, next_state_vector, done))
    
    
    return transitions, total_reward, initial_region, next_region, next_state_coord, final_done, steps