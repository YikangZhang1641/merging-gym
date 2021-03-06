import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import random
import pygame
from pygame.locals import *
import pygame.math as pygame_math
from shapely.geometry import box
from shapely.geometry import Polygon
import math
from helper import mpc_1d

# 应该要由当前观测S获得op未来规划的belief， 其中基于op预测的belief是由数据得来

# font = cv2.FONT_HERSHEY_COMPLEX_SMALL

R = 30000
H, W = 1000, 300
WINDOW_H, WINDOW_W = 1000, 300
dT = 0.2

# reward设计
RFirst = 2.0
RSecond = 1.0
RCollision = -10
vel_penalty = 0.001
time_penalty = 0

# 车道线绝对坐标
TRAJ = np.array([int( R - np.sqrt(R * R - (H - x) * (H - x) )) for x in range(H)])
START_POINT = 50
END_POINT = H - 50

# 车辆碰撞体积
VEHICLE_W, VEHICLE_H = 4, 8

# mpc预测时域
prediction_t = 3.0

# 可视化缩放比例
scale = 5.0

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

collision_bc = 8
tmp = []
for i in range(collision_bc):
    for j in range(collision_bc // 2):
        # if i**2 + j**2 < collision_bc**2:
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
        self.r1_accumulate = 0
        self.r2_accumulate = 0

        pygame.init()
        self.screen = pygame.display.set_mode((3*WINDOW_W, WINDOW_H))
        self.screen.fill((0, 0, 0))

        self.left_screen = pygame.Surface((WINDOW_W, WINDOW_H))
        self.left_screen.fill((255, 255, 255))

        self.right_screen = pygame.Surface((WINDOW_W, WINDOW_H))
        self.right_screen.fill((255, 255, 255))

        pygame.display.set_caption('On ramp merging experiment')

        self.font = pygame.font.Font(None, 17)
        self.mark_font = pygame.font.SysFont(None, 50)

        self.ego = pygame.surfarray.make_surface(np.ones([VEHICLE_W, VEHICLE_H]) * 255)
        self.opponent = pygame.surfarray.make_surface(np.ones([VEHICLE_W, VEHICLE_H]) * 255)

        # action_dict设计，未来t秒后经过的delta v
        self.action_dict = {0: 0, 1:10, 2:20, 3:30, 4:40}
        self.action_space = spaces.Discrete(len(self.action_dict.items()),)
        self.action1 = 1
        self.action2 = 1

        # Create a canvas to render the environment images upon
        self.canvas = np.ones((H, W, 3)) * 1
        self.image = pygame.surfarray.make_surface(self.canvas.transpose(1,0,2) * 255)

        self.time_stamp = 0
        self.reset()
        
        print('MergeEnvd Environment initialized')

    def show_reward(self):
        return RFirst, RSecond, RCollision, vel_penalty

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

    def action_to_acc(self, x0, v0, xt, vt, t):
        res = mpc_1d(x0, v0, xt, vt, t)
        return res.action()

    def step(self, action1, action2=None):
        self.action1, self.action2 = action1, action2

        self.time_stamp += dT
        if self.time_stamp > 500:
            self.done = True
        info = {"collision": False}

        # kinodynamic equations:
        self.state1['acc'] = self.action_to_acc(x0=self.state1['pos'], v0=self.state1['vel'], xt=self.state1['pos'] + self.action_dict[action1] * prediction_t, vt=self.action_dict[action1], t=prediction_t)

        self.state1['vel'] = max(0, self.state1['vel'] + self.state1['acc'] * dT)
        self.state1['pos'] += self.state1['vel'] * dT

        self.state2['acc'] = 0 if action2 is None else self.action_to_acc(x0=self.state2['pos'], v0=self.state2['vel'], xt=self.state2['pos'] + self.action_dict[action2] * prediction_t, vt=self.action_dict[action2], t=prediction_t)
        self.state2['vel'] = max(0, self.state2['vel'] + self.state2['acc'] * dT)
        self.state2['pos'] += self.state2['vel'] * dT

        obs = self.observe()

        reward1 = - time_penalty - vel_penalty * np.abs(self.state1['vel'] - 20.0) # (self.state1['vel'] - 20.0)
        reward2 = - time_penalty - vel_penalty * np.abs(self.state2['vel'] - 20.0) #* (self.state2['vel'] - 20.0)
        # reward1 = -vel_penalty * self.state1['acc'] * self.state1['acc']
        # reward2 = -vel_penalty * self.state2['acc'] * self.state2['acc']

        if self.state1['pos'] > END_POINT:
            if self.winner is None:
                self.winner = 1
                reward1 += RFirst
            elif self.winner is 1:
                reward1 = 0
            else:
                reward1 += RSecond
                self.done = True

        if self.state2['pos'] >= END_POINT:
            if self.winner is None:
                self.winner = 2
                reward2 += RFirst
            elif self.winner is 2:
                reward2 = 0
            else:
                reward2 += RSecond
                self.done = True

        if self.is_collided():
            self.done = True
            reward1 += RCollision
            reward2 += RCollision
            info["collision"] = True

        rewards = [reward1, reward2]

        self.r1_accumulate += reward1
        self.r2_accumulate += reward2

        # print('MergeEnv Step successful!')
        return obs, rewards, self.done, info


    def is_collided(self):
        x1, y1 = lon2coord(self.state1['pos'], "ego")
        x2, y2 = lon2coord(self.state2['pos'], "opponent")
        p1 = Polygon([(p.x, p.y) for p in self.corners(self.ego, x1, y1, 0)])
        p2 = Polygon([(p.x, p.y) for p in self.corners(self.opponent, x2, y2, 0)])
        if p1.intersects(p2):
            print("collided!")
            return True
        return False

    def reset(self):

        start_diff = 50.0
        self.done = False
        self.winner = None
        self.time_stamp = 0

        # 起始条件相同
        self.state1 = {'pos': START_POINT, 'vel': 20.0, 'acc': 0.0}
        self.state2 = {'pos': START_POINT, 'vel': 20.0, 'acc': 0.0}

        # 起始条件随机
        # self.state1 = {'pos': START_POINT + np.random.randn() * 5, 'vel': 20.0 + np.random.randn() * 3, 'acc': 0.0}
        # self.state2 = {'pos': START_POINT + np.random.uniform(-VEHICLE_H/2, VEHICLE_H/2), 'vel': 20.0 + np.random.uniform(-5, 10), 'acc': 0.0}

        self.r1_accumulate = 0
        self.r2_accumulate = 0
        
        # print('MergeEnv Environment reset')
        states = self.observe()
        # print("states:", states)

        return states

    def corners(self, agent, y, x, yaw, scale=1.0):
        rect = agent.get_rect(center=(x, y))
        pivot = pygame_math.Vector2(x, y)
        p0 = scale * (pygame.math.Vector2(rect.topleft) - pivot).rotate(-yaw) + pivot
        p1 = scale * (pygame.math.Vector2(rect.topright) - pivot).rotate(-yaw) + pivot
        p2 = scale * (pygame.math.Vector2(rect.bottomright) - pivot).rotate(-yaw) + pivot
        p3 = scale * (pygame.math.Vector2(rect.bottomleft) - pivot).rotate(-yaw) + pivot
        return p0, p1, p2, p3

    def render(self, goal = None, goal_op = None, player=1, sum_r1 = 0, sum_r2 = 0, tag_left=None, tag_right=None, last_r1=0, last_r2=0):

        x1, y1 = lon2coord(self.state1['pos'], "ego")
        x2, y2 = lon2coord(self.state2['pos'], "opponent")
        x1_t, y1_t = lon2coord(self.state1['pos'] + self.state1['vel'] * prediction_t, "ego")
        x2_t, y2_t = lon2coord(self.state2['pos'] + self.state2['vel'] * prediction_t, "opponent")

        self.left_screen.blit(self.image, (0,0))
        self.right_screen.blit(self.image, (0,0))

        pygame.draw.circle(self.left_screen, color=(0,0,0), center=(scale * (W/2 - R - y2) + WINDOW_W / 2, -scale * x2 + WINDOW_H / 2), radius=scale * (R + VEHICLE_W), width=1)
        pygame.draw.circle(self.left_screen, color=(0,0,0), center=(scale * (W/2 + R - y2) + WINDOW_W / 2, -scale * x2 + WINDOW_H / 2), radius=scale * (R + VEHICLE_W), width=1)
        pygame.draw.circle(self.left_screen, color=(0,0,0), center=(scale * (W/2 - R - y2) + WINDOW_W / 2, -scale * x2 + WINDOW_H / 2), radius=scale * (R - VEHICLE_W), width=1)
        pygame.draw.circle(self.left_screen, color=(0,0,0), center=(scale * (W/2 + R - y2) + WINDOW_W / 2, -scale * x2 + WINDOW_H / 2), radius=scale * (R - VEHICLE_W), width=1)

        pygame.draw.circle(self.right_screen, color=(0,0,0), center=(scale * (W/2 - R - y1) + WINDOW_W / 2, -scale * x1 + WINDOW_H / 2), radius=scale * (R + VEHICLE_W), width=1)
        pygame.draw.circle(self.right_screen, color=(0,0,0), center=(scale * (W/2 + R - y1) + WINDOW_W / 2, -scale * x1 + WINDOW_H / 2), radius=scale * (R + VEHICLE_W), width=1)
        pygame.draw.circle(self.right_screen, color=(0,0,0), center=(scale * (W/2 - R - y1) + WINDOW_W / 2, -scale * x1 + WINDOW_H / 2), radius=scale * (R - VEHICLE_W), width=1)
        pygame.draw.circle(self.right_screen, color=(0,0,0), center=(scale * (W/2 + R - y1) + WINDOW_W / 2, -scale * x1 + WINDOW_H / 2), radius=scale * (R - VEHICLE_W), width=1)


        clr1 = [0,0,0]
        if goal is not None:
            if goal == 0:
                clr1 = [255, 0, 0]
            elif goal == 1:
                clr1 = [0, 0, 255]
        else:
            if self.state1['acc'] > 1e-2:
                clr1 = [255, 0, 0]
            elif self.state1['acc'] < -1e-2:
                clr1 = [0, 0, 255]

        clr2 = [0, 0, 0]
        if goal_op is not None:
            if goal_op == 0:
                clr2 = [255, 0, 0]
            elif goal_op == 1:
                clr2 = [0, 0, 255]
        else:
            if self.state2['acc'] > 1e-2:
                clr2 = [255, 0, 0]
            elif self.state2['acc'] < -1e-2:
                clr2 = [0, 0, 255]


        pygame.draw.polygon(self.left_screen, [120,120,120], self.corners(self.opponent, scale * (x2_t - x2) + 3*WINDOW_H/5, scale * (y2_t - y2) + WINDOW_W/2, yaw=0, scale=scale)[:2] + self.corners(self.opponent, 3*WINDOW_H/5, WINDOW_W/2, yaw=0, scale=scale)[2:], width=0)
        pygame.draw.polygon(self.right_screen, [120,120,120], self.corners(self.ego, scale*(x1_t-x1)+3*WINDOW_H/5, scale*(y1_t-y1)+WINDOW_W/2, yaw=0, scale=scale)[:2] + self.corners(self.ego, 3*WINDOW_H/5, WINDOW_W/2, yaw=0, scale=scale)[2:], width=0)

        pygame.draw.polygon(self.left_screen, [0,0,0], self.corners(self.ego, scale * (x1 - x2) + 3*WINDOW_H/5, scale * (y1 - y2) + WINDOW_W/2, yaw=0, scale=scale), width=0)
        pygame.draw.polygon(self.right_screen, clr1, self.corners(self.ego, scale * (x1 - x1) + 3*WINDOW_H/5, y1 - y1 + WINDOW_W/2, yaw=0, scale=scale), width=0)

        pygame.draw.polygon(self.left_screen, clr2, self.corners(self.opponent, scale * (x2 - x2) + 3*WINDOW_H/5, scale * (y2 - y2) + WINDOW_W/2, yaw=0, scale=scale), width=0)
        pygame.draw.polygon(self.right_screen, [0,0,0], self.corners(self.opponent, scale * (x2 - x1) + 3*WINDOW_H/5, scale * (y2 - y1) + WINDOW_W/2, yaw=0, scale=scale), width=0)



        # if goal is not None:
        #     if goal == 0:
        #         # self.canvas[H - 2 * START_POINT + 0 : H - 2 * START_POINT + 14, W - START_POINT - 0 : W - START_POINT + 14, :] = [0, 0, 1]
        #         pygame.draw.polygon(self.right_screen, [0, 0, 255], [(0.8*WINDOW_W, 0.8*WINDOW_H-10), (0.8*WINDOW_W + 10, 0.8*WINDOW_H-10), (0.8*WINDOW_W+10, 0.8*WINDOW_H), (0.8*WINDOW_W, 0.8*WINDOW_H), ], width=0)
        #
        #     elif goal == 1:
        #         # self.canvas[H - 2 * START_POINT + 14 : H - 2 * START_POINT + 28, W - START_POINT - 0 : W - START_POINT + 14, :] = [0.5, 0, 0.5]
        #         pygame.draw.polygon(self.right_screen, [128, 0, 128], [(0.8*WINDOW_W, 0.8*WINDOW_H), (0.8*WINDOW_W + 10, 0.8*WINDOW_H), (0.8*WINDOW_W+10, 0.8*WINDOW_H+10), (0.8*WINDOW_W, 0.8*WINDOW_H+10), ], width=0)
        #     else:
        #         # self.canvas[H - 2 * START_POINT + 28 : H - 2 * START_POINT + 42, W - START_POINT - 0 : W - START_POINT + 14, :] = [1, 0, 0]
        #         pygame.draw.polygon(self.right_screen, [255, 0, 0], [(0.8*WINDOW_W, 0.8*WINDOW_H+10), (0.8*WINDOW_W + 10, 0.8*WINDOW_H+10), (0.8*WINDOW_W+10, 0.8*WINDOW_H+20), (0.8*WINDOW_W, 0.8*WINDOW_H+20), ], width=0)
        #
        # if goal_op is not None:
        #     if goal_op == 0:
        #         # self.canvas[H - 2 * START_POINT + 0 : H - 2 * START_POINT + 14, W - START_POINT - 14 : W - START_POINT + 0, :] = [0, 0, 1]
        #         pygame.draw.polygon(self.left_screen, [0, 0, 255], [(0.8*WINDOW_W, 0.8*WINDOW_H-10), (0.8*WINDOW_W + 10, 0.8*WINDOW_H-10), (0.8*WINDOW_W+10, 0.8*WINDOW_H), (0.8*WINDOW_W, 0.8*WINDOW_H), ], width=0)
        #     elif goal_op == 1:
        #         # self.canvas[H - 2 * START_POINT + 14 : H - 2 * START_POINT + 28, W - START_POINT - 14 : W - START_POINT + 0, :] = [0.5, 0, 0.5]
        #         pygame.draw.polygon(self.left_screen, [128, 0, 128], [(0.8*WINDOW_W, 0.8*WINDOW_H), (0.8*WINDOW_W + 10, 0.8*WINDOW_H), (0.8*WINDOW_W+10, 0.8*WINDOW_H+10), (0.8*WINDOW_W, 0.8*WINDOW_H+10), ], width=0)
        #     else:
        #         # self.canvas[H - 2 * START_POINT + 28 : H - 2 * START_POINT + 42, W - START_POINT - 14 : W - START_POINT + 0, :] = [1, 0, 0]
        #         pygame.draw.polygon(self.left_screen, [255, 0, 0], [(0.8*WINDOW_W, 0.8*WINDOW_H+10), (0.8*WINDOW_W + 10, 0.8*WINDOW_H+10), (0.8*WINDOW_W+10, 0.8*WINDOW_H+20), (0.8*WINDOW_W, 0.8*WINDOW_H+20), ], width=0)

        spd_l = self.font.render("Spd: " + str(round(self.state2['vel'], 2)), 2, (0,0,0))
        self.left_screen.blit(spd_l, (0.2*WINDOW_W, 0.6*WINDOW_H))
        r2_l = self.font.render("Rwd:" + str(round(self.r2_accumulate, 2)), 2, (0,0,0))
        self.left_screen.blit(r2_l, (0.2*WINDOW_W, 0.6*WINDOW_H + 15))
        # self.left_screen.blit(self.font.render("LastR:" + str(round(last_r2, 2)), 2, (0,0,0)), (0.2*WINDOW_W, 0.6*WINDOW_H + 30))
        # self.left_screen.blit(self.font.render("Score:" + str(round(sum_r2, 2)), 2, (0,0,0)), (0.2*WINDOW_W, 0.6*WINDOW_H + 45))

        if tag_left:
            self.left_screen.blit(self.mark_font.render(tag_left, 2, (0,0,0)), (0.2*WINDOW_W, 0.1*WINDOW_H))

        spd_r = self.font.render("Spd: " + str(round(self.state1['vel'], 2)), 2, (0,0,0))
        self.right_screen.blit(spd_r, (0.7*WINDOW_W, 0.6*WINDOW_H))
        r1_r = self.font.render("Rwd:" + str(round(self.r1_accumulate, 2)), 2, (0,0,0))
        self.right_screen.blit(r1_r, (0.7*WINDOW_W, 0.6*WINDOW_H + 15))
        # self.right_screen.blit(self.font.render("LastR:" + str(round(last_r1, 2)), 2, (0,0,0)), (0.7 * WINDOW_W, 0.6 * WINDOW_H + 30))
        # self.right_screen.blit(self.font.render("Score:" + str(round(sum_r1, 2)), 2, (0,0,0)), (0.7*WINDOW_W, 0.6*WINDOW_H + 45))
        if tag_right:
            self.right_screen.blit(self.mark_font.render(tag_right, 2, (0,0,0)), (0.2*WINDOW_W, 0.1*WINDOW_H))

        self.plot(player)

        pygame.time.wait(50)



    def plot(self, player=1):
        if player == 1:
            self.screen.blit(self.right_screen, (WINDOW_W, 0))
        elif player == 2:
            self.screen.blit(self.left_screen, (0, 0))
            self.screen.blit(self.right_screen, (2 * WINDOW_W, 0))
        pygame.display.update()


    def intro(self, player=1):
        # IMQ & training
        self.left_screen.blit(self.image, (0,0))
        self.right_screen.blit(self.image, (0,0))
        self.plot(player)
        pygame.time.wait(1000)

        #Intro
        self.left_screen.blit(self.font.render("Please pass the ramp quickly without collision", 2, (0, 0, 0)), (0.1 * WINDOW_W, 3*WINDOW_H/5))
        self.right_screen.blit(self.font.render("Please pass the ramp quickly without collision", 2, (0, 0, 0)), (0.1 * WINDOW_W, 3*WINDOW_H/5))
        self.plot(player)
        pygame.time.wait(3000)

    def prepare(self, player=1):
        #Fixation
        self.left_screen.blit(self.image, (0, 0))
        self.right_screen.blit(self.image, (0, 0))
        pygame.draw.lines(self.left_screen,  (0, 0, 0), True, [(0.5 * WINDOW_W - 10, 3*WINDOW_H/5),      (0.5 * WINDOW_W + 10, 3*WINDOW_H/5     )], 3)
        pygame.draw.lines(self.left_screen,  (0, 0, 0), True, [(0.5 * WINDOW_W,      3*WINDOW_H/5 - 10), (0.5 * WINDOW_W,      3*WINDOW_H/5 + 10)], 3)
        pygame.draw.lines(self.right_screen, (0, 0, 0), True, [(0.5 * WINDOW_W - 10, 3*WINDOW_H/5),      (0.5 * WINDOW_W + 10, 3*WINDOW_H/5     )], 3)
        pygame.draw.lines(self.right_screen, (0, 0, 0), True, [(0.5 * WINDOW_W,      3*WINDOW_H/5 - 10), (0.5 * WINDOW_W,      3*WINDOW_H/5 + 10)], 3)
        self.plot(player)
        pygame.time.wait(int(np.random.uniform(1000, 3000)))


    def feedback(self, player=1):
        self.left_screen.blit(self.image, (0, 0))
        self.right_screen.blit(self.image, (0, 0))
        self.left_screen.blit(self.font.render("You earn " + str(round(self.r2_accumulate, 2)) + " points", 2, (0, 0, 0)), (0.3 * WINDOW_W, 3*WINDOW_H/5))
        self.right_screen.blit(self.font.render("You earn " + str(round(self.r1_accumulate, 2)) + " points", 2, (0, 0, 0)), (0.3 * WINDOW_W, 3*WINDOW_H/5))

        self.plot(player)
        pygame.time.wait(3000)

    def finish(self, sum_r1, sum_r2, player=1):
        self.left_screen.blit(self.image, (0, 0))
        self.right_screen.blit(self.image, (0, 0))
        self.left_screen.blit(self.font.render("Games completed. Reward: " + str(round(sum_r2, 2)), 2, (0, 0, 0)), (0.2 * WINDOW_W, 3*WINDOW_H/5))
        self.right_screen.blit(self.font.render("Games completed. Reward: " + str(round(sum_r1, 2)), 2, (0, 0, 0)), (0.2 * WINDOW_W, 3*WINDOW_H/5))
        self.plot(player)
        pygame.time.wait(10000)


    def close(self):
        cv2.destroyAllWindows()

        


class MergeEnvExtend(gym.Env):
    def __init__(self):
        print('MergeEnvExtend Environment initialized')
    def step(self):
        print('MergeEnvExtend Step successful!')
    def reset(self):
        print('MergeEnvExtend Environment reset')
