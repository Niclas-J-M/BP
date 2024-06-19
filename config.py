class Config:
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.995  # Initial exploration rate
    epsilon_min = 0.005  # Minimum exploration rate
    epsilon_decay = 0.95  # Decay rate per episode