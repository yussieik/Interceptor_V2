#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:39:31 2019

@author: yussiroz
"""
import gym
import interceptor
from collections import deque
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam  


class DQNAgent():
    def __init__(self, env_id, path, episodes, max_env_steps, win_threshold, epsilon_decay,
                 state_size=None, action_size=None, epsilon=1.0, epsilon_min=0.01, 
                 gamma=1, alpha=.01, alpha_decay=.01, batch_size=16, prints=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_id)
 
        if state_size is None: 
            self.state_size = np.ravel(self.env.observation_space)
        else: 
            self.state_size = state_size
 
        if action_size is None: 
            self.action_size = len(self.env._action_set)
        else: 
            self.action_size = action_size
 
        self.episodes = episodes
        self.env._max_episode_steps = max_env_steps
        self.win_threshold = win_threshold
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.path = path                     #location where the model is saved to
        self.prints = prints                 #if true, the agent will print his scores
 
        self.model = self._build_model()
  
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size.shape[0], activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        return model
    
    def act(self, state):
        if (np.random.random() <= self.epsilon):
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))   
                         
                         
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
RockeAgent = DQNAgent('Interceptor-v2', './', 3, 1000, 1.0, 0.9)        
        
Environment = RockeAgent.env
np.random.seed(14)
table=list()
state = Environment.observation_space
table.append(state)

for stp in range(40):
    #action_button = np.random.randint(0,4)                
    action_button = RockeAgent.act(np.ravel(state))        
    #--> agent(state) --> a
    state, reward = Environment.step(action_button)
#    n_img = Environment.Draw()
    table.append([state,action_button,reward])

        