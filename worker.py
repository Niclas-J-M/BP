import torch
import torch.optim as optim
from networks import PolicyNetwork, ValueNetwork
from utils.prioritized_memory import Memory


class Worker:
    def __init__(self, input_dim, output_dim, gamma = 0.99):
        self.policy_net = PolicyNetwork(input_dim, output_dim)
        self.value_net = ValueNetwork(input_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0007)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=0.0007)
        self.memory = Memory(5000)  # Capacity
        self.batch_size = 256
        self.gamma = gamma
        self.value_loss_weight = 0.01
        self.policy_losses = []
        self.value_losses = []
        self.entropy_coeff = 0.01

    def train(self, transitions):
        """ On-policy training method using collected transitions """
        for state, action, reward, next_state, done in transitions:
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            action = torch.tensor([action], dtype=torch.long)
            reward = torch.tensor([reward], dtype=torch.float32)

            # Predict the current state's value and the next state's value
            predicted_value = self.value_net(state)
            next_value = self.value_net(next_state)

            # Calculate advantage and target value for the critic
            td_target = reward + self.gamma * next_value * (1 - int(done))
            td_error = td_target - predicted_value

            # Value network update
            value_loss = td_error.pow(2).mean()
            if torch.isnan(value_loss):
                print(f"NaN in value loss: {value_loss}")
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            # Policy network update
            log_prob = torch.log(self.policy_net(state)[action])
            policy_loss = -log_prob * td_error.detach()  # detach to stop gradients for the policy update
            if torch.isnan(policy_loss):
                print(f"NaN in policy loss: {policy_loss}")
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Store experiences in the replay buffer for SIL
            self.memory.add(td_error.item(), (state.numpy(), action.item(), reward.item(), next_state.numpy(), done))

    def train_sil2(self):
        """ Self-Imitation Learning from the memory buffer """
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to perform a batch update

        transitions, idxs, is_weights = self.memory.sample(self.batch_size)
        for (state, action, reward, next_state, done), idx, is_weight in zip(transitions, idxs, is_weights):
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            action = torch.tensor([action], dtype=torch.long)
            reward = torch.tensor([reward], dtype=torch.float32)

            # Policy network update
            log_prob = torch.log(self.policy_net(state)[action])
            policy_loss = -log_prob * reward  # Directly use the reward as a surrogate for advantage

            policy_loss = policy_loss * torch.tensor([is_weight], dtype=torch.float32)  # Apply importance sampling weights
            if torch.isnan(policy_loss):
                print(f"NaN in policy loss (SIL): {policy_loss}")


            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Update priorities in the memory based on the new policy
            new_td_error = reward - self.value_net(state)  # New error based on current policy
            if torch.isnan(new_td_error).any():
                print(f"NaN in new td error (SIL): {new_td_error}")
            self.memory.update([idx], [new_td_error.item()])


    def train_sil(self):
        """ Self-Imitation Learning from the memory buffer """
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to perform a batch update

        transitions, idxs, is_weights = self.memory.sample(self.batch_size)
        epsilon = 1e-8  # Small value to add for numerical stability

        for (state, action, reward, next_state, done), idx, is_weight in zip(transitions, idxs, is_weights):
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            action = torch.tensor([action], dtype=torch.long)
            reward = torch.tensor([reward], dtype=torch.float32)
            is_weight = torch.tensor([is_weight], dtype=torch.float32)


            # Policy network update
            probs = self.policy_net(state)
            log_prob = torch.log(probs[action] + epsilon)
            policy_loss = -log_prob * reward  # Directly use the reward as a surrogate for advantage
            policy_loss = policy_loss * torch.tensor([is_weight], dtype=torch.float32)  # Apply importance sampling weights
            
            # Debugging statements
            if torch.isnan(log_prob).any():
                print(f"NaN in log_prob: {log_prob}")
            if torch.isnan(policy_loss).any():
                print(f"NaN in policy_loss: {policy_loss}")
            if torch.isnan(reward).any():
                print(f"NaN in reward: {reward}")
            if torch.isnan(is_weight).any():
                print(f"NaN in is_weight: {is_weight}")

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Update priorities in the memory based on the new policy
            new_td_error = reward - self.value_net(state)  # New error based on current policy
            
            # Debugging statements
            if torch.isnan(new_td_error).any():
                print(f"NaN in new_td_error: {new_td_error}")
            
            self.memory.update([idx], [new_td_error.item()])




    def select_action2(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            probs = self.policy_net(state)
            print(f"Action probabilities: {probs}") 
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()
    
    def select_action3(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            probs = self.policy_net(state)
            if torch.isnan(probs).any():
                print(f"NaN in probs: {probs}")
                raise ValueError(f"NaN in probs: {probs}")

            # Cap probabilities to avoid very low and very high values
            min_prob = 1e-8
            max_prob = 1.0 - 1e-8
            probs = torch.clamp(probs, min=min_prob, max=max_prob)
            probs = probs / probs.sum()  # Re-normalize to ensure they sum to 1
            
            if torch.isnan(probs).any():
                print(f"NaN after clamping in probs: {probs}")
                raise ValueError(f"NaN after clamping in probs: {probs}")

        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            probs = self.policy_net(state)
            if torch.isnan(probs).any():
                print(f"NaN in probs: {probs}")
                raise ValueError(f"NaN in probs: {probs}")

        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()
