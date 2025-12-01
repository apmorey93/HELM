import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgObsWrapper

print("Minigrid version:", minigrid.__version__)
print("Gymnasium version:", gym.__version__)

try:
    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    print("Default obs shape:", env.reset()[0]['image'].shape)
    
    from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper
    
    env_full = FullyObsWrapper(env)
    env_full = ImgObsWrapper(env_full)
    print("FullyObsWrapper shape:", env_full.reset()[0].shape)
    
    env_rgb = RGBImgObsWrapper(env)
    env_rgb = ImgObsWrapper(env_rgb)
    print("RGBImgObsWrapper shape:", env_rgb.reset()[0].shape)
except Exception as e:
    print("Failed:", e)
