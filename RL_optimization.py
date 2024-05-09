import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from CA2D import CA2D
from creature import Creature, Encyclopedia
from biodiversity_analysis import creature_search, run_decompos, recenter_creatures

# Define your custom environment class inheriting from gym.Env
class RLEnv(gym.Env):
    def __init__(self, n_actions, H, W, patch_size, zoom_ratio, max_frame):
        super(RLEnv, self).__init__()
        
        # self.action_space = gym.spaces.Discrete(n_actions) 
        self.action_space = gym.spaces.MultiDiscrete([n_actions]*2)
        self.observation_space = gym.spaces.Box(low=0, high=255,shape=(H,W))
        self.patch_size = patch_size
        self.zoom_ratio = zoom_ratio
        self.max_frame = max_frame
        self.encyclopedia = Encyclopedia(patch_size = self.patch_size, zoom_ratio = self.zoom_ratio)
        self.H = H
        self.W = W
        self.auto = CA2D((self.H, self.W), s_num='23', b_num='3', random=True)
        # Variables for reward components
        self.creature_num  = []
        self.creature_diversity = []
        self.encyclopedia_size = []
        
        self.ws_buff = []
        self.frame_count = 0
    
    def reset(self):
        # Reset environment to initial state
        self.auto.reset()
        self.frame_count = 0
        self.encyclopedia = Encyclopedia(patch_size = self.patch_size, zoom_ratio = self.zoom_ratio)
        self.creature_num = []
        self.creature_diversity = []
        self.encyclopedia_size = []
        self.ws_buff = []
        return self.auto.world
    
    def step(self, action):
        # Take action in the environment
        # Update environment state
        self.update_rules(action)
        self.auto.step()
        self.frame_count += 1
        self.auto.draw()
        self.ws_buff.append(self.auto.worldmap)
        
        if self.frame_count%3:
            self.update_reward_components()
            self.encyclopedia.update_tracks()
            self.ws_buff = []
        
        # Calculate reward
        reward = self.calculate_reward()
        # Check if the episode is done
        done = False
        if self.frame_count > self.max_frame:
            done = True 
        # Return observation, reward, done, info
        return self.auto.world, reward, done, {}
    
    def when_done(self): #TODO Implement graph plots when episode is done 
        raise NotImplementedError()
    
    def update_rules(self, action): 
        self.auto.increment_rules(action[0], action[1])
    
    def update_reward_components(self):
        # Update reward components
        creatures, loc_arr, loc_x, loc_y = creature_search(
                self.ws_buff, self.W, self.H, patch_size=self.patch_size, halved=True
            )
        if len(creatures) > 2:
            centered_creaures = recenter_creatures(
                creatures, self.patch_size, zoom_ratio = self.zoom_ratio
            )
            temp = self.encyclopedia.get_num_creatures()
            for i, creature in enumerate(centered_creaures):
                c = Creature(creature, creature_pos = [loc_x[i], loc_y[i]])
                self.encyclopedia.update(c)
            self.creature_num.append(self.encyclopedia.get_num_creatures() - temp)
            self.creature_diversity.append([len(track) for track in self.encyclopedia.tracks])
            
    def calculate_reward(self):
        # Calculate reward
        if len(self.creature_diversity):
            alpha = 0.4 #TODO find a good way to normalize this
            r1 = np.sum(np.array(self.creature_diversity[-1]) > 0, axis = 0)
            r2 = self.encyclopedia.get_num_creatures()
            reward =  r1 + r2*alpha
        else:
            reward = 0 
        return reward

RL_dir = "RL_experiments"
os.makedirs(RL_dir, exist_ok=True)
max_frame= 50
# Initialize your custom environment
env = RLEnv(n_actions=4, H=400, W=300, patch_size=8, zoom_ratio=2, max_frame = max_frame)

# Wrap the environment in a vectorized environment
env = DummyVecEnv([lambda: env])

# Define and train the PPO agent
model = PPO("MlpPolicy",
            env,
            verbose=1,
            n_steps = 2048,
            batch_size = 64,
            n_epochs = 10,
            learning_rate = 0.0001,
            )
model.learn(total_timesteps=max_frame)

# Save the trained model
model.save(os.path.join(RL_dir, "ppo_custom_env"))

# Load the trained model
# model = PPO.load("ppo_custom_env")

# Evaluate the trained model
obs = env.reset()
for _ in range(1):
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(reward)
        if done:
            obs = env.reset()
        

# Close the environment
env.close()

#TODO Add video visualization of the results
#TODO Discover how to extract the right set of rules
#TODO Play with the reward function and find regularization terms
