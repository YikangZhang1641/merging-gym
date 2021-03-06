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



OP_MODEL = "dqn"# 定义用的对手车模型
player = 2 if OP_MODEL == "pvp" else 1


def main():
    if not os.path.isdir("log"):
        os.mkdir("log")
    log_path = os.path.join("log", datetime.datetime.now().strftime("%Y--%m--%d %H:%M:%S"))
    os.mkdir(log_path)

    episodes = 5
    collision_count = 0

    if OP_MODEL == "hdqn":
        load_path = "2022--03--31 00:01:27 hdqn 对手L0初始随机, acc强制0，对手到达也不停车(2.0, 1.0, -100, 0.001)"
        upper_op = Goal_DQN(load_path)
        lower_op = HDQN(load_path)
        goal, goal_op = None, None
        print("using model: hdqn")

    elif OP_MODEL == "dqn":
        # load_path = "2022--03--30 18:48:33normal dqn(2.0, 1.0, -10, 0.01)" #for test, 可视为L0
        #########################
        load_path = "2022--03--31 03:37:35normal dqn with OP:L0(2.0, 1.0, -10, 0.001)" # L1: 平衡，保持在20
        # load_path = "2022--03--31 21:36:59normal dqn with OP:L1(2.0, 1.0, -10, 0.001)" # L2: 基本匀速20
        # load_path = "2022--03--31 20:37:39normal dqn with OP:L0(2.0, 1.0, -10, 0.001)" # L1: 激进
        # load_path = "2022--03--31 14:45:59normal dqn with OP:L1(2.0, 1.0, -10, 0.001)" # L2: 激进，35左右
        #########################
        # load_path = "2022--03--31 21:33:10normal dqn with OP:L2(2.0, 1.0, -10, 0.001)" # L3: 激进策略


        dqn = DQN(load_path)
        print("using model: dqn")

    elif OP_MODEL == "rainbow_dqn":
        current_model = RainbowDQN(env.observation_space.shape[0], env.action_space.n, num_atoms, Vmin, Vmax)
        if load_path is not None:
            current_model.load_state_dict(torch.load(os.path.join(load_path, "eval.pth")))

    else:
        load_path = "pvp"
        print("using model: pvp")

    sum_r1, sum_r2 = 0, 0
    last_r1, last_r2 = 0, 0

    env.intro(player)
    for i in range(episodes):
        state = env.reset()
        next_state = state
        ep_reward = 0
        done = False
        action = 2
        action_op = 2

        # env.render(last_r1=last_r1, last_r2=last_r2, sum_r1=sum_r1, sum_r2=sum_r2, tag_left="3", tag_right="3")
        # pygame.time.wait(1000)
        # env.render(last_r1=last_r1, last_r2=last_r2, sum_r1=sum_r1, sum_r2=sum_r2, tag_left="2", tag_right="2")
        # pygame.time.wait(1000)
        # env.render(last_r1=last_r1, last_r2=last_r2, sum_r1=sum_r1, sum_r2=sum_r2, tag_left="1", tag_right="1")
        # pygame.time.wait(1000)
        env.prepare(player=player)

        filename = os.path.join(log_path, "episode" + str(i) + " " + load_path)
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["x2 - x1", "y2 - y1", "self.state2['vel'] - self.state1['vel']", "END_POINT - self.state1['pos']", "self.state1['vel']", "x1 - x2", "y1 - y2", "self.state1['vel'] - self.state2['vel']", "END_POINT - self.state2['pos']", "self.state2['vel']", "action1", "action2", "reward1", "reward2"])
            while not done:
                if env.winner is None:
                    tag_left, tag_right = None, None
                elif env.winner == 1:
                    tag_left, tag_right = None, "Finished"
                else:
                    tag_left, tag_right = "Finished", None

                env.render(last_r1=last_r1, last_r2=last_r2, sum_r1=sum_r1, sum_r2=sum_r2, tag_left=tag_left, tag_right=tag_right, player=player)

                key_pressed = pygame.key.get_pressed()
                # print(sum(key_pressed))

                ########### ego command ###########
                if key_pressed[pygame.K_KP0]:
                    action = 0
                elif key_pressed[pygame.K_KP1]:
                    action = 1
                elif key_pressed[pygame.K_KP2]:
                    action = 2
                elif key_pressed[pygame.K_KP3]:
                    action = 3
                elif key_pressed[pygame.K_KP4]:
                    action = 4

                # if key_pressed[pygame.K_UP]:
                #     action += 1
                #     if action > NUM_ACTIONS - 1:
                #         action = NUM_ACTIONS - 1
                #     print("ego UP")
                #
                # elif key_pressed[pygame.K_DOWN]:
                #     action -= 1
                #     if action < 0:
                #         action = 0
                #     print("ego Down")

                ########## opponent command ###########
                if OP_MODEL == "hdqn":
                    if goal_op is None or goal_op == goal_status( op_state(state) ):
                        goal_op = upper_op.choose_goal( op_state(state) )

                    goal_state_op = torch.unsqueeze(torch.FloatTensor([goal_op] + op_state(state)), dim=0)
                    action_op = lower_op.choose_action(goal_state_op)

                elif OP_MODEL == "dqn":
                    action_op = dqn.choose_action( op_state(state) )

                elif OP_MODEL == "rainbow_dqn":
                    action_op = current_model.act( op_state(state) )

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

                if env.winner is not 1:
                    writer.writerow(state + [action, action_op] + rewards)

                if info["collision"]:
                    collision_count += 1
                    print("Collided!", collision_count / (i + 1))

                state = next_state

        sum_r1 += env.r1_accumulate
        sum_r2 += env.r2_accumulate
        env.render(last_r1=last_r1, last_r2=last_r2, sum_r1=sum_r1, sum_r2=sum_r2, tag_left="Finished", tag_right="Finished", player=player)
        last_r1 = env.r1_accumulate
        last_r2 = env.r2_accumulate

        pygame.time.wait(1000)

        env.feedback(player=player)
    env.finish(sum_r1=sum_r1, sum_r2=sum_r2, player=player)


if __name__ == '__main__':
    main()
