from gymnasium.envs.registration import register
from .core import DetentBottleSim, VIEWS, WRISTS
from .gym_env import DetentBottleEnv

register(id='DetentBottle-v0', entry_point='detent_bottle.gym_env:DetentBottleEnv', max_episode_steps=None)
