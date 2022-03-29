import gym
import merging_gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import datetime
from tensorboardX import SummaryWriter
import pygame
import csv

USE_CUDA = torch.cuda.is_available()

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.7
MEMORY_CAPACITY = 2000
GOAL_MEMORY_CAPACITY = 200

Q_NETWORK_ITERATION = 100

env = gym.make("merging_env-v0")
# env = gym.make("CartPole-v0")

env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_GOALS = 3
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

from hdqn import HDQN, Goal_DQN, goal_status
from main import DQN
from ranbowdqn import RainbowDQN

def op_state(state):
    return state[NUM_STATES // 2:] + state[:NUM_STATES // 2]



OP_MODEL = "hdqn"# 定义用的对手车模型


def main():
    if not os.path.isdir("log"):
        os.mkdir("log")
    log_path = os.path.join("log", datetime.datetime.now().strftime("%Y--%m--%d %H:%M:%S"))
    os.mkdir(log_path)

    episodes = 100
    collision_count = 0

    if OP_MODEL == "hdqn":
        load_path = "2022--03--29 14:05:36"
        upper_op = Goal_DQN(load_path)
        lower_op = HDQN(load_path)
        goal, goal_op = None, None
        print("using model: hdqn")

    elif OP_MODEL == "dqn":
        load_path = "l1"
        dqn = DQN(load_path)
        print("using model: dqn")

    elif OP_MODEL == "rainbow_dqn":
        current_model = RainbowDQN(env.observation_space.shape[0], env.action_space.n, num_atoms, Vmin, Vmax)
        if load_path is not None:
            current_model.load_state_dict(torch.load(os.path.join(load_path, "eval.pth")))

    else:
        print("using model: pvp")


    for i in range(episodes):
        state = env.reset()
        next_state = state
        ep_reward = 0
        done = False
        action = 2
        action_op = 2

        filename = os.path.join(log_path, "episode"+str(i))
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["x2 - x1", "y2 - y1", "self.state2['vel'] - self.state1['vel']", "END_POINT - self.state1['pos']", "self.state1['vel']", "x1 - x2", "y1 - y2", "self.state1['vel'] - self.state2['vel']", "END_POINT - self.state2['pos']", "self.state2['vel']", "action1", "action2", "reward1", "reward2"])
            while not done:
                env.render()

                key_pressed = pygame.key.get_pressed()
                # print(sum(key_pressed))

                ########### ego command ###########
                if key_pressed[pygame.K_UP]:
                    action += 1
                    if action > NUM_ACTIONS - 1:
                        action = NUM_ACTIONS - 1
                    print("ego UP")

                elif key_pressed[pygame.K_DOWN]:
                    action -= 1
                    if action < 0:
                        action = 0
                    print("ego Down")

                ########## opponent command ###########
                if OP_MODEL == "hdqn":
                    if goal_op is None or goal_op == goal_status( op_state(state) ):
                        goal_op = upper_op.choose_goal( op_state(state) )

                    goal_state_op = torch.unsqueeze(torch.FloatTensor([goal_op] + op_state(state)), dim=0)
                    action_op = lower_op.choose_action(goal_state_op)

                elif OP_MODEL == "dqn":
                    action_op = dqn.choose_action(state)

                elif OP_MODEL == "rainbow_dqn":
                    action_op = current_model.act(state[3:] + state[:3])

                else:
                    if key_pressed[pygame.K_w]:
                        action_op += 1
                        if action_op > NUM_ACTIONS - 1:
                            action_op = NUM_ACTIONS - 1
                        print("opponent UP")

                    elif key_pressed[pygame.K_s]:
                        action_op -= 1
                        if action_op < 0:
                            action_op = 0
                        print("opponent Down")
                ###################################

                pygame.event.pump()
                next_state, rewards, done, info = env.step(action, action_op)

                writer.writerow(state + [action, action_op] + rewards)

                if info["collision"]:
                    collision_count += 1
                    print("Collided!", collision_count / (i + 1))

                state = next_state


if __name__ == '__main__':
    main()
