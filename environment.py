from gymnasium.envs.registration import register
import gymnasium as gym

# Register custom environments
def register_env():
    # register(
    #     id='SimpleEnv2-v0',
    #     entry_point='simple_env2:SimpleEnv2',
    # )

    register(
        id='SimpleEnv-v0',
        entry_point='simple_env:SimpleEnv',
    )