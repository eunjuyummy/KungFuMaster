from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C

env = make_atari_env('ALE/KungFuMaster-v5')
env = VecFrameStack(env, n_stack=4)

model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1000000000)
model.save("KungFuMaster")

del model

model = A2C.load("KungFuMaster")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    env.render(mode="human")
