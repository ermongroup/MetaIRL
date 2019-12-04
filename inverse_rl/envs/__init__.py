import logging

from gym.envs import register

LOGGER = logging.getLogger(__name__)

_REGISTERED = False
def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering custom gym environments")
    register(id='PointMazeRight-v0', entry_point='inverse_rl.envs.point_maze_env:PointMazeEnv',
             kwargs={'sparse_reward': False, 'direction': 1, 'discrete': True})
    register(id='PointMazeLeft-v0', entry_point='inverse_rl.envs.point_maze_env:PointMazeEnv',
             kwargs={'sparse_reward': False, 'direction': 0, 'discrete': True})
    register(id='PointMazeRightCont-v0', entry_point='inverse_rl.envs.point_maze_env:PointMazeEnv',
             kwargs={'sparse_reward': False, 'direction': 1, 'discrete': False})
    register(id='PointMazeLeftCont-v0', entry_point='inverse_rl.envs.point_maze_env:PointMazeEnv',
             kwargs={'sparse_reward': False, 'direction': 0, 'discrete': False})

