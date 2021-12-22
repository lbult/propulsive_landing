import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from RL_PL import SRM_PL_RL

#make directories
log_path = os.path.join("Training", "Logs")
PPO_load = os.path.join("Training", "Saved Models", "RL_PL1")
PPO_save = os.path.join("Training", "Saved Models", "RL_PL2")

env = DummyVecEnv([lambda: SRM_PL_RL()])
#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

model = PPO.load(PPO_load, env=env)
model.learn(total_timesteps=200000)
model.save(PPO_save)

# TEST Model
episodes = 40
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action, _ = model.predict(obs) #NOW USING MODEL HERE
        obs, reward, done, info = env.step(action)
        env.render()
        score += reward
    print("Episode:{} Score:{}".format(episode, score))

env.reset()
#env.close()