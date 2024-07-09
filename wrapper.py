

#=======================================================================#

class NESWActionsImage(Wrapper):
    """
    we change the actions to be 0:N, 1:E, 2:S, 3:W
    """

    def __init__(self, env, stocastic_action_prob=0, max_num_actions=100):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(5)
        self.key = 0
        self.door = 0
        self.counter_of_steps = 0
        self.stocastic_action_prob = stocastic_action_prob
        self.n_actions = 4
        self.unwrapped.max_steps = 1e10
        self.one_hot_dim = self.unwrapped.grid.width * self.unwrapped.grid.height
        self.encoding_pos = np.arange(self.one_hot_dim).reshape((self.unwrapped.grid.width, self.unwrapped.grid.height))
        self.max_steps = max_num_actions

    def one_hot_encode(self, pos):
        one_hot = np.zeros(self.one_hot_dim)
        one_hot[self.encoding_pos[pos[0], pos[1]]] = 1
        return one_hot

    def reset(self, **kwargs):
        self.counter_of_steps = 0
        self.key=0
        self.door=0
        obs, info = self.env.reset(**kwargs)
        self.env.unwrapped.gen_obs()

        env = self.unwrapped
        x, y = tuple(env.agent_pos)
        image = env.get_frame()

        one_hot = self.one_hot_encode((x,y))

        return {"pos": (x, y, self.key, self.door), "image":image, "one_hot": one_hot}
    
    def step(self, action):
        # Variables to manage stochastic action probabilities and actual actions taken
        if random.random() < self.stocastic_action_prob:
            action = random.choice(list(range(self.n_actions)))

        obs, reward, done, truncated, info = None, 0, False, False, {}

        # Align the agent's direction based on the action
        if action < 4:  # Assuming actions 0-3 are directional moves
            desired_dir = (3 - action) % 4  # Calculate desired direction based on action
            # Rotate agent to face the correct direction
            while self.unwrapped.agent_dir != desired_dir:
                self.env.step(0)  # Rotate agent

            # Check the contents of the cell in front of the agent
            wall = False
            fwd_pos = self.unwrapped.front_pos
            fwd_cell = self.unwrapped.grid.get(*fwd_pos)
            if fwd_cell is not None:
                if fwd_cell.type == 'wall':
                    wall = True
            if fwd_cell is not None:
                if fwd_cell.type == 'key':
                    self.key = 1
                    self.env.step(3)  # Pick up the key
                elif fwd_cell.type == 'door' and self.door == 0 and self.key == 1:
                    self.door = 1
                    self.env.step(5)  # Open the door
                    
            obs, reward, done, truncated, info = self.env.step(2)  # Move forward
            if wall:
                reward = -0.1

            # Move forward if not a wall 
        elif action == 4:  # Null action, no movement
            obs, reward, done, truncated, info = self.env.step(0)

        # Update position and gather environment frame for return
        x, y = tuple(self.unwrapped.agent_pos)
        image = self.unwrapped.get_frame()
        self.counter_of_steps += 1
        if self.counter_of_steps > self.max_steps:
            done = True

        # Manage keys and doors for reward logic
        if self.key == 1 and self.door == 1:
            reward = 1
            done = False
            self.key = 0  # Reset key state if needed

        one_hot = self.one_hot_encode((x, y))
        return {"pos": (x, y, self.key, self.door), "image": image, "one_hot": one_hot}, reward, done, info


    def step2(self, action):

        if random.random() < self.stocastic_action_prob:
            action =  random.choice(list(range(self.n_actions)))

        if action == 0:
            dir = self.unwrapped.agent_dir
            while dir != 3: # set the correct direction
                self.env.step(0)
                dir = self.unwrapped.agent_dir

            # Get the position in front of the agent
            fwd_pos = self.unwrapped.front_pos
            # Get the contents of the cell in front of the agent
            fwd_cell = self.unwrapped.grid.get(*fwd_pos)
            if fwd_cell is not None:
                if fwd_cell.type == 'key':
                    self.key = 1
                    self.env.step(3)  # pickup the key

                elif fwd_cell.type == 'door' and self.door == 0 and self.key == 1:
                    self.door = 1
                    self.env.step(5)  # open the door

            obs, reward, done, truncated, info = self.env.step(2)  # move forward

        elif action == 1:
            dir = self.unwrapped.agent_dir
            while dir != 0: # set the correct direction
                self.env.step(0)
                dir = self.unwrapped.agent_dir

            # Get the position in front of the agent
            fwd_pos = self.unwrapped.front_pos
            # Get the contents of the cell in front of the agent
            fwd_cell = self.unwrapped.grid.get(*fwd_pos)
            if fwd_cell is not None:
                if fwd_cell.type == 'key':
                    self.key = 1
                    self.env.step(3)  # pickup the key

                elif fwd_cell.type == 'door' and self.door == 0 and self.key == 1:
                    self.door = 1
                    self.env.step(5)  # open the door

            obs, reward, done, truncated, info = self.env.step(2)  # move forward

        elif action == 2:
            dir = self.unwrapped.agent_dir
            while dir != 1: # set the correct direction
                self.env.step(0)
                dir = self.unwrapped.agent_dir

            # Get the position in front of the agent
            fwd_pos = self.unwrapped.front_pos
            # Get the contents of the cell in front of the agent
            fwd_cell = self.unwrapped.grid.get(*fwd_pos)
            if fwd_cell is not None:
                if fwd_cell.type == 'key':
                    self.key = 1
                    self.env.step(3)  # pickup the key

                elif fwd_cell.type == 'door' and self.door == 0 and self.key == 1:
                    self.door = 1
                    self.env.step(5)  # open the door

            obs, reward, done, truncated, info = self.env.step(2)  # move forward

        elif action == 3:
            dir = self.unwrapped.agent_dir
            while dir != 2: # set the correct direction
                self.env.step(0)
                dir = self.unwrapped.agent_dir

            # Get the position in front of the agent
            fwd_pos = self.unwrapped.front_pos
            # Get the contents of the cell in front of the agent
            fwd_cell = self.unwrapped.grid.get(*fwd_pos)
            if fwd_cell is not None:
                if fwd_cell.type == 'key':
                    self.key = 1
                    self.env.step(3)  # pickup the key

                elif fwd_cell.type == 'door' and self.door == 0 and self.key == 1:
                    self.door = 1
                    self.env.step(5)  # open the door
            obs, reward, done, truncated, info = self.env.step(2)  # move forward

        elif action == 4: # Null action
            obs, reward, done, truncated, info = self.env.step(0)

        x, y = tuple(self.unwrapped.agent_pos)
        image = self.unwrapped.get_frame()
        self.counter_of_steps += 1
        if self.counter_of_steps > self.max_steps:
            done = True

        if self.key == 1 and self.door == 1:
            reward = 1
            done = False
            self.key = 0

        one_hot = self.one_hot_encode((x,y))

        return {"pos": (x, y, self.key, self.door), "image":image, "one_hot": one_hot}, reward, done, info

    def get_goal_position(self):
        return self.unwrapped.goal_position