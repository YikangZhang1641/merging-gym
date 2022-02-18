import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import random

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

R = 30000
H, W = 1000, 200
dT = 1.0
param = 1
RFirst = 500
RSecond = 300
RCollision = -1000
ACC = 2.0

def lon2coord(lon, id):
    angle = np.arctan2( H, R ) - lon / R
    # print(lon, (R-W) / R, np.arccos( (R-W) / R ), lon / R)
    x = R * np.sin(angle)
    if id is "ego":
        y = W/2 + (R - R * np.cos(angle))
    elif id is "opponent":
        y = W/2 - (R - R * np.cos(angle))
    else:
        print("wrong character")
    return x, y

TRAJ = np.array([int( R - np.sqrt(R*R - (H - x) * (H - x) )) for x in range(H)])
START_POINT = 50.0
END_POINT = H - 50.0

collision_bc = 8
tmp = []
for i in range(collision_bc):
    for j in range(collision_bc):
        if i**2 + j**2 < collision_bc**2:
            tmp.append([i,j])
            tmp.append([-i,j])
            tmp.append([i,-j])
            tmp.append([-i,-j])
BC = np.array(tmp)



class MergeEnv(gym.Env):
    def __init__(self):
        super(MergeEnv, self).__init__()
        self.observation_shape = (6) #p,v,a for 1; p,v,a for 2)
        self.observation_space = spaces.Box(low = np.array([0, 0, -10, 0, 0, -10]), 
                                            high = np.array([H, 100, 10, H, 100, 10]),
                                            dtype = np.float16)

        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(3,)
        # self.Vexp_action = {0: 0, 1: 10, 2: 20, 3:30, 4:40}
        self.Vexp_action = {0: 20, 1: 40, 2: 60}

        # Create a canvas to render the environment images upon 
        self.canvas = np.ones((H, W, 3)) * 1
        self.canvas[[H-1-i for i in range(H)], W // 2 + TRAJ, :] = 0
        self.canvas[[H-1-i for i in range(H)], W // 2 - TRAJ, :] = 0

        self.reset()

        # Define elements present inside the environment
        # self.elements = []
        
        # Maximum fuel chopper can take at once
        # self.max_fuel = 1000

        # Permissible area of helicper to be 
        # self.y_min = int (self.observation_shape[0] * 0.1)
        # self.x_min = 0
        # self.y_max = int (self.observation_shape[0] * 0.9)
        # self.x_max = self.observation_shape[1]
        
        print('MergeEnvd Environment initialized')


    def step(self, action1, action2):

        info = {"collision": False}

        # kinodynamic equations:
        Vexp1 = self.Vexp_action[action1]
        if self.state1['vel'] < Vexp1:
            self.state1['acc'] = ACC 
        elif self.state1['vel'] > Vexp1:
            self.state1['acc'] = -ACC 
        else:
            self.state1['acc'] = 0 

        self.state1['vel'] += self.state1['acc'] * dT
        self.state1['pos'] += self.state1['vel'] * dT
        
        Vexp2 = self.Vexp_action[action2]
        if self.state2['vel'] < Vexp2:
            self.state2['acc'] = ACC
        elif self.state2['vel'] > Vexp2:
            self.state2['acc'] = -ACC
        else:
            self.state2['acc'] = 0

        self.state2['vel'] += self.state2['acc'] * dT
        self.state2['pos'] += self.state2['vel'] * dT

        obs = [self.state1['pos'], self.state1['vel'], self.state1['acc'], self.state2['pos'], self.state2['vel'], self.state2['acc']]

        # reward1 = -param * (self.state1['vel'] - 20.0) * (self.state1['vel'] - 20.0)
        # reward2 = -param * (self.state2['vel'] - 20.0) * (self.state2['vel'] - 20.0)
        reward1 = -param * self.state1['acc'] * self.state1['acc']
        reward2 = -param * self.state2['acc'] * self.state2['acc']

        if not self.done and self.state1['pos'] > END_POINT:
            self.done = True
            reward1 += RFirst
            reward2 += RSecond

        if not self.done and self.state2['pos'] >= END_POINT:
            self.done = True
            reward2 += RFirst
            reward1 += RSecond

        
        if self.vehicle_distance() <= 2 * collision_bc:
            self.done = True
            reward1 += RCollision
            reward2 += RCollision
            info["collision"] = True
            return obs, [reward1, reward2], self.done, info

        rewards = [reward1, reward2]
        

        # print('MergeEnv Step successful!')
        return obs, rewards, self.done, info

    def vehicle_distance(self):
        x1, y1 = lon2coord(self.state1['pos'], "ego")
        x2, y2 = lon2coord(self.state2['pos'], "opponent")
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

    def reset(self):
        start_diff = 50.0
        self.done = False
        self.winner = None

        # self.state1 = {'pos': START_POINT + int(np.random.rand() * start_diff), 'vel': 20.0 + np.random.random_integers(10), 'acc': 0.0}
        # self.state2 = {'pos': START_POINT + int(np.random.rand() * start_diff), 'vel': 20.0 + np.random.random_integers(10), 'acc': 0.0}
        self.state1 = {'pos': START_POINT, 'vel': 20.0, 'acc': 0.0}
        self.state2 = {'pos': START_POINT, 'vel': 20.0, 'acc': 0.0}
        
        
        # print('MergeEnv Environment reset')
        states = [self.state1['pos'], self.state1['vel'], self.state1['acc'], self.state2['pos'], self.state2['vel'], self.state2['acc']]
        # print("states:", states)

        return states

    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            canvas_bak = self.canvas.copy()

            # x1 = H -1 - int(self.state1['pos'])
            # y1 = W//2 + TRAJ[H -1-x1]
            # print("x1,y1", x1, y1)

            x1, y1 = lon2coord(self.state1['pos'], "ego")
            # print("x1,y1", x1, y1)

            # self.canvas[x1+1-collision_bc : x1+collision_bc, y1+1-collision_bc : y1+collision_bc, :] = [0,0,1]
            self.canvas[int(x1)+BC[:, 0], int(y1)+BC[:, 1], :] = [0,0,1]

            # x2 = H -1 - int(self.state2['pos'])
            # y2 = W//2 - TRAJ[H -1-x2]
            # print("x2,y2", x2, y2)

            x2, y2 = lon2coord(self.state2['pos'], "opponent")
            # print("x2,y2", x2, y2)
            # self.canvas[x2+1-collision_bc : x2+collision_bc, y2+1-collision_bc : y2+collision_bc, :] = [1,0,0]
            self.canvas[int(x2)+BC[:, 0], int(y2)+BC[:, 1], :] = [1,0,0]

            cv2.imshow("Game", self.canvas)
            cv2.waitKey(100)
            self.canvas = canvas_bak

        elif mode == "rgb_array":
            return self.canvas
        
    def close(self):
        cv2.destroyAllWindows()

        


class MergeEnvExtend(gym.Env):
    def __init__(self):
        print('MergeEnvExtend Environment initialized')
    def step(self):
        print('MergeEnvExtend Step successful!')
    def reset(self):
        print('MergeEnvExtend Environment reset')
