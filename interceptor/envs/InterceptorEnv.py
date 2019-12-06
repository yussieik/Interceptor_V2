import gym
from gym import error, spaces, utils
from gym.utils import seeding
import check_real_time
import numpy as np



class InterceptorEnv(gym.Env):  
    metadata = {'render.modes': ['human']}   
    
    def __init__(self):
#        Define action space {0,1,2,3}
        self._action_set = [0,1,2,3]
        self.action_space = spaces.Discrete(len(self._action_set))
#       Define observation space/state
        self.observation_space = np.zeros((192,386)).astype('uint8')
        self.state = self.observation_space
        self.current_step = 0
        self.done = False
        check_real_time.Init()

        
    def step(self, action):
        reward = 0.0        
#        for stp in range(1000):
        action = self._action_set[action] 
        r_locs, i_locs, c_locs, ang, score = check_real_time.Game_step(action)
        n_img=check_real_time.Draw()
        state=check_real_time.Create_state(n_img)
        self.state = state
        reward = score
        self.current_step += 1       
        return state, reward
            
    def reset(self):
        self.state = self.observation_space
        self.current_step = 0
        self.done = False        
        

    def render(self, mode='human', close=False):
        pass







