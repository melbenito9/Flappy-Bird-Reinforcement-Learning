'''
#Game environment for SARSA.
'''

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

import gym
from gym.wrappers import Monitor
import gym_ple
from gym_ple import PLEEnv
from utils import *

import warnings
warnings.filterwarnings('ignore')


class FBSarsa(gym.Wrapper):
    
    def __init__(self, env, rounding = None):
        
        super().__init__(env)
        self.rounding = rounding
    
    def save_output(self, outdir = None):
      
        if outdir:
            self.env = Monitor(self.env, directory = outdir, force = True)
        
    def step(self, action):
        
        _, reward, terminal, _ = self.env.step(action)
        state = self.getGameState()
        if not terminal: reward += 0.1
        else: reward = -100
        if reward >= 1: reward = 5
        return state, reward, terminal, {}

    def getGameState(self):
        
        gameState = self.env.game_state.getGameState()
        hor_dist_to_next_pipe = gameState['next_pipe_dist_to_player']
        ver_dist_to_next_pipe = gameState['next_pipe_bottom_y'] - gameState['player_y']
        if self.rounding:
            hor_dist_to_next_pipe = discretize(hor_dist_to_next_pipe, self.rounding)
            ver_dist_to_next_pipe = discretize(ver_dist_to_next_pipe, self.rounding)
            
        state = []
        state.append('player_vel' + ' ' + str(gameState['player_vel']))
        state.append('hor_dist_to_next_pipe' + ' ' + str(hor_dist_to_next_pipe))
        state.append('ver_dist_to_next_pipe' + ' ' + str(ver_dist_to_next_pipe))
        return ' '.join(state)
        


class Transform(object):
    ''' A class that preprocesses the images of the game screen. '''
    
    def __init__(self):
        ''' Initializes the class. '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transform = transforms.Compose([
              transforms.Lambda(lambda x: x[:404]),
              transforms.ToPILImage(),
              transforms.Grayscale(),
              transforms.Resize((80, 80)),
              transforms.Lambda(lambda x:
                    cv2.threshold(np.array(x), 128, 255, cv2.THRESH_BINARY)[1]),
              transforms.ToTensor(),
              transforms.Lambda(lambda x: x.unsqueeze(0).to(self.device)),
        ])

    def process(self, img):
        return self.transform(img)