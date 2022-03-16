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


def op_state(state):
    return state[NUM_STATES // 2:] + state[:NUM_STATES // 2]



OP_MODEL = "dqn"# 定义用的对手车模型


def main():
    episodes = 10
    print("Collecting Experience....")
    reward_list = []
    q_eval_list = []
    collision_list = []
    winner_list = []
    # plt.ion()
    fig, ax = plt.subplots(4, 1)
    ax[0].set_xlim(0, episodes)
    ax[1].set_xlim(0, episodes)
    ax[2].set_xlim(0, episodes)
    ax[3].set_xlim(0, episodes)
    collision_count = 0
    win_count = 0

    if OP_MODEL == "hdqn":
        load_path = "2022--03--08 14:08:51"
        upper_op = Goal_DQN(load_path)
        lower_op = HDQN(load_path)
        goal, goal_op = None, None
    elif OP_MODEL == "dqn":
        load_path = None
        dqn = DQN(load_path)


    for i in range(episodes):
        state = env.reset()
        next_state = state
        ep_reward = 0
        done = False

        while not done:
            env.render()

            if OP_MODEL == "hdqn":
                ########## hdqn ############
                if goal_op is None or goal_op == goal_status( op_state(state) ):
                    goal_op = upper_op.choose_goal( op_state(state) )

                goal_state_op = torch.unsqueeze(torch.FloatTensor([goal_op] + op_state(state)), dim=0)
                action_op = lower_op.choose_action(goal_state_op)
                ############################

            elif OP_MODEL == "dqn":
                ########## dqn #############
                action_op = dqn.choose_action(state)
                ############################

            action = 1
            key_pressed = pygame.key.get_pressed()
            print(sum(key_pressed))

            if key_pressed[pygame.K_UP]:
                action = 2
                print("UP")

            elif key_pressed[pygame.K_DOWN]:
                action = 0
                print("Down")

            pygame.event.pump()
            next_state, rewards, done, info = env.step(action, action_op)

            if info["collision"]:
                collision_count += 1
                print("Collided!", collision_count / (i + 1))

            state = next_state

if __name__ == '__main__':
    main()
