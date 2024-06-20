import torch
from worker_head import Worker_Head
from manager_head import Manager_Head

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_dim = 4
output_dim = 4
tasks = [0, 1, 2]  # Example tasks
manager = Manager_Head(input_dim, output_dim, tasks, device)

initial_region = 1
goal_region1 = 2
goal_region2 = 3
goal_region3 = 101

task = tasks[0]


# Parameters for the test
input_shape = 9  # Example input dimension
output_shape = 4  # Example output dimension
batch_size = 5  # Example batch size

manager.add_region(initial_region, task)

if initial_region not in manager.Q[task] and initial_region < input_dim + 1:
    manager.add_region(initial_region, task)
if goal_region1 not in manager.Q[task] and goal_region1 < input_dim + 1:
    manager.add_region(goal_region1, task)

if goal_region1 > input_dim + 1:
    print("task option")
    worker = manager.get_task_specific_worker(initial_region, goal_region1, input_shape, output_shape, task)
else:
    print("region option")
    worker = manager.get_create_region_option(initial_region, input_shape, output_shape)

manager.add_option(initial_region, goal_region1, task)

option_idx = worker.add_option_head()
print("Added first option head with index:", option_idx)
if initial_region < input_dim + 1 and goal_region1 < input_dim + 1:
    for n_task in tasks:
        manager.option_indices[n_task][initial_region][goal_region1] = option_idx
else:
    manager.option_indices[task][initial_region][goal_region1] = option_idx



#===================================================================

if initial_region not in manager.Q[task] and initial_region < input_dim + 1:
    manager.add_region(initial_region, task)
if goal_region2 not in manager.Q[task] and goal_region2 < input_dim + 1:
    manager.add_region(goal_region2, task)

if goal_region2 > input_dim + 1:
    print("task option")
    worker = manager.get_task_specific_worker(initial_region, goal_region2, input_shape, output_shape, task)
else:
    print("region option")
    worker = manager.get_create_region_option(initial_region, input_shape, output_shape)

manager.add_option(initial_region, goal_region2, task)

option_idx = worker.add_option_head()
print("Added first option head with index:", option_idx)
if initial_region < input_dim + 1 and goal_region2 < input_dim + 1:
    for n_task in tasks:
        manager.option_indices[n_task][initial_region][goal_region2] = option_idx
else:
    manager.option_indices[task][initial_region][goal_region2] = option_idx


#===================================================================

if initial_region not in manager.Q[task] and initial_region < input_dim + 1:
    manager.add_region(initial_region, task)
if goal_region3 not in manager.Q[task] and goal_region3 < input_dim + 1:
    manager.add_region(goal_region3, task)

if goal_region3 > input_dim + 1:
    print("task option")
    worker = manager.get_task_specific_worker(initial_region, goal_region3, input_shape, output_shape, task)
else:
    print("region option")
    worker = manager.get_create_region_option(initial_region, input_shape, output_shape)

manager.add_option(initial_region, goal_region3, task)
option_idx = worker.add_option_head()
print("Added first option head with index:", option_idx)
if initial_region < input_dim + 1 and goal_region3 < input_dim + 1:
    for n_task in tasks:
        manager.option_indices[n_task][initial_region][goal_region3] = option_idx
else:
    manager.option_indices[task][initial_region][goal_region3] = option_idx

# print("==================================")
# print(manager.Q)
# print("==================================")
# print(manager.option_indices)
# print("==================================")

actual_end_region = None

if manager.options_in_state_space(initial_region, task):
    goal_region = manager.select_action(initial_region, task)
    print("goal", goal_region)
    # task option
    if goal_region > input_dim:
        # print("task option")
        worker = manager.get_task_specific_worker(initial_region, goal_region, input_shape, output_shape, task)
    else:
        # print("region option")
        worker = manager.get_create_region_option(initial_region, input_shape, output_shape)

    actual_end_region = 103

    if actual_end_region != goal_region and actual_end_region != initial_region:
        print("act != goal")
        if actual_end_region not in manager.Q[task] and actual_end_region < input_dim + 1:
            manager.add_region(actual_end_region, task)
        if actual_end_region not in manager.Q[task][initial_region]:
            if actual_end_region > input_dim:
                worker = manager.get_task_specific_worker(initial_region, actual_end_region, input_shape, output_shape, task)
            else:
                worker = manager.get_create_region_option(initial_region, input_shape, output_shape)
            # print("actual not in manager.Q")
            manager.add_option(initial_region, actual_end_region, task)
            option_idx = worker.add_option_head()
            if initial_region < input_dim + 1 and actual_end_region < input_dim + 1:
                for n_task in tasks:
                    manager.option_indices[n_task][initial_region][actual_end_region] = option_idx
            else:
                manager.option_indices[task][initial_region][actual_end_region] = option_idx
        else:
            if actual_end_region > input_dim:
                # print("task option")
                worker = manager.get_task_specific_worker(initial_region, actual_end_region, input_shape, output_shape, task)
            else:
                # print("region option")
                worker = manager.get_create_region_option(initial_region, input_shape, output_shape)


print("actual", actual_end_region)
print("==================================")
print(manager.Q)
print("==================================")
print(manager.option_indices)
print("==================================")

print(initial_region)
print(actual_end_region)

goal_region = actual_end_region

if goal_region > input_dim:
    option_idx = 0
else:
    option_idx = manager.option_indices[task][initial_region][goal_region]

print("Option_idx", option_idx)


option = goal_region
# The task option
if goal_region > input_dim:
    goal_region = initial_region
# The task option has been found 
if actual_end_region > input_dim and actual_end_region != 0:
    option = actual_end_region
    goal_region = initial_region

print("option", option)
print("goal_region", goal_region)
print("initial_region", initial_region)



print(manager.general_workers)
print("===============Task specific================================")
print(manager.task_specific_workers)

print(worker)

# if goal_region2 > input_dim + 1:
#     print("task option")
#     worker = manager.get_task_specific_worker(initial_region, goal_region1, input_shape, output_shape, task)
# else:
#     print("region option")
#     worker = manager.get_create_region_option(initial_region, input_shape, output_shape)


# input_shape2 = (9, )

# states = torch.zeros(batch_size, *input_shape2).to(device)
# for i in range(batch_size):
#     state_index = torch.randint(0, input_shape2[0], (1,)).item()  # Random state index within 9
#     states[i, state_index] = 1


# #print(states)
# # Perform forward pass with the first head
# policy_output1 = worker.policy_net(states, option_idx1)
# value_output1 = worker.value_net(states, option_idx1)

# print("Policy output with head {}:".format(option_idx1), policy_output1)
# print("Value output with head {}:".format(option_idx1), value_output1)

# # # Perform forward pass with the second head
# policy_output2 = worker.policy_net(states, option_idx2)
# value_output2 = worker.value_net(states, option_idx2)

# print("Policy output with head {}:".format(option_idx2), policy_output2)
# print("Value output with head {}:".format(option_idx2), value_output2)

# # # Try selecting actions using the first head
# actions1 = [worker.select_action(state.cpu().numpy(), option_idx1) for state in states]
# print("Selected actions with head {}:".format(option_idx1), actions1)

# # Try selecting actions using the second head
# actions2 = [worker.select_action(state.cpu().numpy(), option_idx2) for state in states]
# print("Selected actions with head {}:".format(option_idx2), actions2)

# # # Create dummy transitions for training
transitions = []
for _ in range(2):
    states = torch.zeros(input_shape)
    state_index = torch.randint(0, input_shape, (1,)).item()  # Random state index within 9
    states[state_index] = 1
    action = torch.randint(0, output_dim, (1,)).item()
    reward = torch.randn(1).item()
    next_state = torch.zeros(input_shape)
    state_index2 = torch.randint(0, input_shape, (1,)).item()  # Random state index within 9
    next_state[state_index2] = 1
    done = torch.randint(0, 2, (1,)).item()
    transitions.append((states, action, reward, next_state, done))

print(transitions)
# #print(transitions)
# # # Train the network using the first head
# worker.train(transitions, option_idx1)
# print("Training with head {} completed.".format(option_idx1))

# # # Train the network using the second head
# worker.train(transitions, option_idx2)
# print("Training with head {} completed.".format(option_idx2))


