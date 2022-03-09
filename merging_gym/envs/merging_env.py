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
dT = 0.2
param = 0.0001
RFirst = 1.0
RSecond = 1.0
RCollision = -10
ACC_INC = 1
ACC_DEC = -2

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
START_POINT = 50
END_POINT = H - 50

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
        self.observation_shape = (10) #dx,dy,dv, x, v for 1; dx,dy,dv,x,v for 2)
        self.observation_space = spaces.Box(low = np.array([-H, -W, -100, 0,    0, -H, -W, -100, 0,   0]),
                                            high = np.array([H,  W,  100, H,  100,  H,  W,  100, H, 100]),
                                            dtype = np.float16)

        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(3,)
        # self.action_dict = {0: ACC_DEC, 1:ACC_DEC/2, 2:0, 3:ACC_INC/2, 4:ACC_INC}
        self.action_dict = {0: ACC_DEC, 1:0, 2:ACC_INC}
        # self.Vexp_action = {0: 0, 1: 10, 2: 20, 3:30, 4:40}
        # self.Vexp_action = {0: 20, 1: 40, 2: 60}

        # Create a canvas to render the environment images upon 
        self.canvas = np.ones((H, W, 3)) * 1
        self.canvas[[H-1-i for i in range(H)], W // 2 + TRAJ, :] = 0
        self.canvas[[H-1-i for i in range(H)], W // 2 - TRAJ, :] = 0
        self.canvas_bak = self.canvas.copy()

        self.time_stamp = 0
        self.reset()
        
        print('MergeEnvd Environment initialized')

    def observe(self):
        x1, y1 = lon2coord(self.state1['pos'], "ego")
        x2, y2 = lon2coord(self.state2['pos'], "opponent")

        obs = [x2 - x1,
               y2 - y1,
               self.state2['vel'] - self.state1['vel'],
               END_POINT - self.state1['pos'],
               self.state1['vel'],
               x1 - x2,
               y1 - y2,
               self.state1['vel'] - self.state2['vel'],
               END_POINT - self.state2['pos'],
               self.state2['vel']]
        return obs

    def step(self, action1, action2):
        self.time_stamp += dT
        if self.time_stamp > 300:
            self.done = True
        info = {"collision": False}

        # kinodynamic equations:
        self.state1['acc'] = self.action_dict[action1]
        # Vexp1 = self.Vexp_action[action1]
        # if self.state1['vel'] < Vexp1:
        #     self.state1['acc'] = ACC_INC
        # elif self.state1['vel'] > Vexp1:
        #     self.state1['acc'] = ACC_DEC
        # else:
        #     self.state1['acc'] = 0 

        self.state1['vel'] = max(0, self.state1['vel'] + self.state1['acc'] * dT)
        self.state1['pos'] += self.state1['vel'] * dT
        
        self.state2['acc'] = self.action_dict[action2]
        # Vexp2 = self.Vexp_action[action2]
        # if self.state2['vel'] < Vexp2:
        #     self.state2['acc'] = ACC_INC
        # elif self.state2['vel'] > Vexp2:
        #     self.state2['acc'] = ACC_DEC
        # else:
        #     self.state2['acc'] = 0

        self.state2['vel'] = max(0, self.state2['vel'] + self.state2['acc'] * dT)
        self.state2['pos'] += self.state2['vel'] * dT

        obs = self.observe()

        reward1 = -param * (self.state1['vel'] - 20.0) * (self.state1['vel'] - 20.0)
        reward2 = -param * (self.state2['vel'] - 20.0) * (self.state2['vel'] - 20.0)
        # reward1 = -param * self.state1['acc'] * self.state1['acc']
        # reward2 = -param * self.state2['acc'] * self.state2['acc']

        if not self.done and self.state1['pos'] > END_POINT:
            self.done = True
            reward1 += RFirst
            reward2 += RSecond

        if not self.done and self.state2['pos'] >= END_POINT:
            self.done = True
            reward2 += RFirst
            reward1 += RSecond

        
        if self.vehicle_distance() <= 2 * collision_bc or self.state1['pos'] < START_POINT or self.state2['pos'] < START_POINT:
            self.done = True
            reward1 += RCollision
            reward2 += RCollision
            info["collision"] = True

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
        self.time_stamp = 0
        # self.state1 = {'pos': START_POINT + int(np.random.rand() * start_diff), 'vel': 20.0 + np.random.random_integers(10), 'acc': 0.0}
        # self.state2 = {'pos': START_POINT + int(np.random.rand() * start_diff), 'vel': 20.0 + np.random.random_integers(10), 'acc': 0.0}
        self.state1 = {'pos': START_POINT + np.random.randn() * 5, 'vel': 20.0 + np.random.randn() * 3, 'acc': 0.0}
        self.state2 = {'pos': START_POINT + np.random.randn() * 5, 'vel': 20.0 + np.random.randn() * 3, 'acc': 0.0}
        
        
        # print('MergeEnv Environment reset')
        states = self.observe()
        # print("states:", states)

        return states

    def render(self, goal = None, goal_op = None):
        mode = "human"
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            # canvas_bak = self.canvas.copy()
            self.canvas = self.canvas_bak.copy()

            x1, y1 = lon2coord(self.state1['pos'], "ego")
            clr = [0,0,0]
            if self.state1['acc'] > 0:
                clr = [0,0,1]
            elif self.state1['acc'] < 0:
                clr = [1,0,0]
            self.canvas[int(x1)+BC[:, 0], int(y1)+BC[:, 1], :] = clr

            x2, y2 = lon2coord(self.state2['pos'], "opponent")
            clr = [0,0,0]
            if self.state2['acc'] > 0:
                clr = [0,0,1]
            elif self.state2['acc'] < 0:
                clr = [1,0,0]
            self.canvas[int(x2)+BC[:, 0], int(y2)+BC[:, 1], :] = clr

            if goal is not None:
                if goal == 0:
                    self.canvas[H - 2 * START_POINT + 0 : H - 2 * START_POINT + 14, W - START_POINT - 0 : W - START_POINT + 14, :] = [0, 0, 1]
                elif goal == 1:
                    self.canvas[H - 2 * START_POINT + 14 : H - 2 * START_POINT + 28, W - START_POINT - 0 : W - START_POINT + 14, :] = [0.5, 0, 0.5]
                else:
                    self.canvas[H - 2 * START_POINT + 28 : H - 2 * START_POINT + 42, W - START_POINT - 0 : W - START_POINT + 14, :] = [1, 0, 0]

            if goal_op is not None:
                if goal_op == 0:
                    self.canvas[H - 2 * START_POINT + 0 : H - 2 * START_POINT + 14, W - START_POINT - 14 : W - START_POINT + 0, :] = [0, 0, 1]
                elif goal_op == 1:
                    self.canvas[H - 2 * START_POINT + 14 : H - 2 * START_POINT + 28, W - START_POINT - 14 : W - START_POINT + 0, :] = [0.5, 0, 0.5]
                else:
                    self.canvas[H - 2 * START_POINT + 28 : H - 2 * START_POINT + 42, W - START_POINT - 14 : W - START_POINT + 0, :] = [1, 0, 0]

            cv2.imshow("Game", self.canvas)
            cv2.waitKey(50)

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
