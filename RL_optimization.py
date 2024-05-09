
import matplotlib.pyplot as plt
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from CA2D import CA2D
from creature import Creature, Encyclopedia
from biodiversity_analysis import creature_search, run_decompos, recenter_creatures, plot_encyclopedia
import pygame
from Camera import Camera
from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.reward_log = []
    
    def _on_step(self) -> bool:
        # Log and display the episode reward
        rewards = self.locals["rewards"]
        self.logger.record("reward", np.mean(rewards))
        self.reward_log.append(np.mean(rewards))
        return True

# Define your custom environment class inheriting from gym.Env
class RLEnv(gym.Env):
    def __init__(self, n_actions, H, W, patch_size, zoom_ratio, max_frame):
        super(RLEnv, self).__init__()
        
        # self.action_space = gym.spaces.Discrete(n_actions) 
        self.action_space = gym.spaces.MultiDiscrete([n_actions]*2)
        self.observation_space = gym.spaces.Box(low=0, high=255,shape=(W,H,3))
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
        self.reward = 0
    
    def reset(self):
        # Reset environment to initial state
        self.auto.change_num(23, 3)
        self.frame_count = 0
        self.encyclopedia = Encyclopedia(patch_size = self.patch_size, zoom_ratio = self.zoom_ratio)
        self.creature_num = []
        self.creature_diversity = []
        self.encyclopedia_size = []
        self.ws_buff = []
        self.reward = 0
        self.update_rate = 50
        self.creature_buffer_size = 3
        return self.auto.worldmap
    
    def step(self, action):
        # Take action in the environment
        # Update environment state
        self.auto.step()
        self.frame_count += 1
        self.auto.draw()
        self.ws_buff.append(self.auto.worldmap)
        
        if self.frame_count%self.creature_buffer_size:
            self.update_reward_components()
            self.encyclopedia.update_tracks()
            self.ws_buff = []
        if self.frame_count%self.update_rate:
            self.update_rules(action)
        
        # Calculate reward
        self.reward += self.calculate_reward()
        # Check if the episode is done
        done = False
        if self.frame_count > self.max_frame:
            # reward = self.calculate_reward()
            done = True 
        # Return observation, reward, done, info
        return self.auto.worldmap, self.reward, done, {"encyclopedia" : self.encyclopedia}
    
    def when_done(self): #TODO Implement graph plots when episode is done 
        raise NotImplementedError()
    
    def update_rules(self, action): 
        self.auto.increment_rules(action[0], action[1])
        # pass
    
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
        r1 = 0
        if len(self.creature_diversity):
            r1 = np.sum(np.array(self.creature_diversity[-1]) > 0, axis = 0)
        r2 = self.encyclopedia.get_num_types()
        r3 = self.encyclopedia.get_num_creatures()
        r4 = self.encyclopedia.get_num_tracks()
        r5 = 10000 if self.auto.worldmap.sum() <= 0.5*self.H*self.W else 0
        p5 = 10000 if self.auto.worldmap.sum() > 0.5*self.H*self.W else 0
        p1 = 900000 if (self.auto.worldmap.sum() == 0 or self.auto.worldmap.sum() == self.H*self.W) else 0
        p4 = self.reward*0.1 if self.reward > 0 else 0
        reward =  r3 - p4 - p1 + r5 
        return reward

def plot_reward(reward_log, save_path):
    plt.plot(reward_log)
    plt.xlabel("Number of steps")
    plt.ylabel("Cumulative reward")
    plt.savefig(save_path)
    plt.close()


RL_dir = "RL_experiments"
os.makedirs(RL_dir, exist_ok=True)
max_frame= 200
H = 300
W = 400
patch_size = 10
# num_env = 5
# Initialize your custom environment
env = RLEnv(n_actions=20, H=H, W=W, patch_size=patch_size, zoom_ratio=2, max_frame = max_frame)

# Wrap the environment in a vectorized environment
env = DummyVecEnv([lambda: env])

# Define and train the PPO agent
model = PPO("MlpPolicy",
            env,
            verbose=1,
            n_steps = 600,
            batch_size = max_frame,
            n_epochs = 100,
            learning_rate = 0.0001,
            
            )
callback = RewardCallback()
# model.learn(total_timesteps=max_frame, callback=callback)
# model.learn(total_timesteps=max_frame)

# Save the trained model
# model.save(os.path.join(RL_dir, "ppo_custom_env"))
# Save the reward log
# plot_reward(callback.reward_log, os.path.join(RL_dir, "reward_log.png"))

# Load the trained model
model = PPO.load(os.path.join(RL_dir, "ppo_custom_env"))

# Evaluate the trained model
obs = env.reset()
done = np.array([False])
pygame.init()
screen = pygame.display.set_mode((W, H), flags=pygame.SCALED | pygame.RESIZABLE)
clock = pygame.time.Clock()
camera = Camera(W,H)
for i in range(10):
    done = np.array([False])
    wait = False
    while not done.all():

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = np.array([True])
            camera.handle_event(event)  # Handle the camera events
            if event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_q):
                    done = np.array([True])
                if (event.key == pygame.K_SPACE):
                    wait = not wait
        if not wait:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            surface = pygame.surfarray.make_surface(obs[0])
            for track_creat in info[0]["encyclopedia"].tracks:
                for track in track_creat:
                    rect = pygame.Rect((track[0], track[1]), (patch_size, patch_size))
                    pygame.draw.rect(surface, (255, 0, 0), rect, 1)

            # text = str(action)
            # font = pygame.font.Font(size = 5)
            # text = font.render(text, True, "green")
            # screen.blit(text, (10, 10))
            
        zoomed_surface = camera.apply(surface)
        screen.blit(zoomed_surface, (0, 0))
        pygame.display.flip()
        clock.tick(90)
        
        if done.all():
            obs = env.reset()
            if info[0]["encyclopedia"].get_num_creatures() > 0:
                plot_encyclopedia(info[0]["encyclopedia"], output_dir=RL_dir, filename=f"encyclopedia_{i}.png")
pygame.quit()
# Close the environment
env.close()

