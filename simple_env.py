from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.envs import register
import gymnasium as gym



UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=13,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        #self.goal_state = goal_state
    
        mission_space = MissionSpace(mission_func=self._gen_mission)

        nA = 5
        nS = (size - 2) * (size - 2)
        self.nrow = nrow = size
        self.ncol = ncol = size 

        self.observation_space = gym.spaces.Discrete(nS)
        self.action_space = gym.spaces.Discrete(nA)

        

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Reach the goal"


    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Add surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.vert_wall(4, 1, 1)
        self.grid.vert_wall(8, 1, 1)
        self.grid.vert_wall(4, 11, 1)
        self.grid.vert_wall(8, 11, 1)

        self.grid.vert_wall(4, 3, 3)
        self.grid.vert_wall(8, 3, 3)

        self.grid.vert_wall(4, 7, 3)
        self.grid.vert_wall(8, 7, 3)

        self.grid.vert_wall(1, 4, 1)
        self.grid.vert_wall(1, 8, 1)
        self.grid.vert_wall(11, 4, 1)
        self.grid.vert_wall(11, 8, 1)

        self.grid.horz_wall(3, 4, 3)
        self.grid.horz_wall(7, 4, 3)

        self.grid.horz_wall(3, 8, 3)
        self.grid.horz_wall(7, 8, 3)
        

        self.put_obj(Goal(), width - 2, height - 2)
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        self.mission = "Reach the goal"