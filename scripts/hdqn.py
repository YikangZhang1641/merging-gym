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


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, num_inputs, num_outputs):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 200)
        self.fc1.weight.data.uniform_(0,1)
        self.fc2 = nn.Linear(200,100)
        self.fc2.weight.data.uniform_(0,1)
        self.out = nn.Linear(100,num_outputs)
        self.out.weight.data.uniform_(0,1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class Goal_DQN():
    """docstring for H-DQN"""

    def __init__(self, load_path=None):
        super(Goal_DQN, self).__init__()
        self.meta_eval_net = Net(NUM_STATES, NUM_GOALS)
        self.meta_target_net = Net(NUM_STATES, NUM_GOALS)
        if USE_CUDA:
            self.meta_eval_net = self.meta_eval_net.cuda()
            self.meta_target_net = self.meta_target_net.cuda()

        if load_path is not None:
            self.meta_eval_net.load_state_dict(torch.load(os.path.join(load_path, "meta_eval.pth")))
            self.meta_target_net.load_state_dict(torch.load(os.path.join(load_path, "meta_target.pth")))

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((GOAL_MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.meta_eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_goal(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array
        state = state.cuda() if USE_CUDA else state

        if np.random.randn() <= EPISILO:  # greedy policy
            goal_value = self.meta_eval_net.forward(state)
            goal = torch.max(goal_value, 1)[1].cpu().numpy()[0]
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:  # random policy
            goal = np.random.randint(0, NUM_GOALS)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        # goal = torch.nn.functional.softmax(goal_value, dim = 1)
        return goal

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % GOAL_MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.meta_target_net.load_state_dict(self.meta_eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch from memory
        sample_index = np.random.choice(GOAL_MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

        if USE_CUDA:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()

        # q_eval
        q_eval = self.meta_eval_net(batch_state).gather(1, batch_action)
        q_next = self.meta_target_net(batch_next_state).detach()
        q_eval4next = self.meta_eval_net(batch_next_state).detach()

        # selected_q_next = q_next.max(1)[0]
        max_act4next = q_eval4next.max(1)[1]
        selected_q_next = q_next[range(128), max_act4next]

        q_target = batch_reward + GAMMA * selected_q_next.view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class HDQN():
    """docstring for H-DQN"""
    def __init__(self, load_path=None):
        super(HDQN, self).__init__()
        self.eval_net = Net(NUM_STATES + 1, NUM_ACTIONS)
        self.target_net = Net(NUM_STATES + 1, NUM_ACTIONS)
        if USE_CUDA:
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
            
        if load_path is not None:
            self.eval_net.load_state_dict(torch.load(os.path.join(load_path, "eval.pth")))
            self.target_net.load_state_dict(torch.load(os.path.join(load_path, "target.pth")))

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, (NUM_STATES + 1) * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        # state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        # goal_state = torch.cat([goal, state], dim=1)
        state = state.cuda() if USE_CUDA else state

        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].cpu().numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = torch.cat((state, torch.FloatTensor([[float(action), reward]]), next_state), dim=1)
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition[0].detach().numpy()
        self.memory_counter += 1

    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES+1])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+2:NUM_STATES+3])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES-1:])

        if USE_CUDA:
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_next_state = batch_next_state.cuda()

            #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_eval4next = self.eval_net(batch_next_state).detach()

        # selected_q_next = q_next.max(1)[0]
        max_act4next = q_eval4next.max(1)[1]
        selected_q_next = q_next[range(128), max_act4next]
        
        q_target = batch_reward + GAMMA * selected_q_next.view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def goal_status(states):
    dx1, dy1, dv1, x1, v1, dx2, dy2, dv2, x2, v2 = states
    ego_time = x1 / (v1 + 0.01)
    # op_time = states[3] / (states[4] + 0.01)
    if ego_time < (x2 - 2.0) / (v2 + 0.01) * 0.9:
        return 0
    elif ego_time < (x2 + 2.0) / (v2 + 0.01) * 1.1:
        return 1
    return 2



def main():

    episodes = 8000
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

    load_path = None
    upper = Goal_DQN(load_path)
    lower = HDQN(load_path)

    upper_op = Goal_DQN(load_path)
    lower_op = HDQN(load_path)

    # opponent = dqn
    # opponent = DQN("results/self")
    # opponent = DQN("L1")
    # opponent2 = DQN("L0")

    goal, goal_op = None, None

    # goal = meta_model.act(state, epsilon_by_frame(frame_idx))
    output_path = datetime.datetime.now().strftime("%Y--%m--%d %H:%M:%S")
    writer = SummaryWriter(log_dir = output_path)

    for i in range(episodes):
        state = env.reset()
        next_state = state
        ep_reward = 0
        done = False
        while not done:

            goal = upper.choose_goal(state)
            # goal_op = 1
            goal_op = upper_op.choose_goal(state[NUM_STATES//2:] + state[:NUM_STATES//2])
            extrinsic_reward = 0

            while not done:
                # env.render(goal, goal_op)

                goal_state = torch.unsqueeze(torch.FloatTensor([goal] + state), dim=0)
                action = lower.choose_action(goal_state)

                goal_state_op = torch.unsqueeze(torch.FloatTensor([goal_op] + state[NUM_STATES//2:] + state[:NUM_STATES//2]), dim=0)
                action_op = lower_op.choose_action(goal_state_op)
                # action_op = (NUM_ACTIONS) // 2

                next_state, rewards, done, info = env.step(action, action_op)
                goal = upper.choose_goal(next_state)
                next_goal_state = torch.unsqueeze(torch.FloatTensor([goal] + next_state), dim=0)
                print("episode", i, "ego/op action,", action, action_op)

                if info["collision"]:
                    collision_count += 1
                    print("Collided!", collision_count / (i+1))

                reward, reward_op = rewards
                ep_reward += reward
                extrinsic_reward += reward
                intrinsic_reward = 1.0 if goal == goal_status(state) else 0.0

                lower.store_transition(goal_state, action, intrinsic_reward, next_goal_state)

                if lower.memory_counter >= MEMORY_CAPACITY and lower.memory_counter:
                    lower.learn()
                state = next_state

                if done or goal == goal_status(state):
                    break

            upper.store_transition(state, goal, extrinsic_reward, next_state)
            if upper.memory_counter >= GOAL_MEMORY_CAPACITY and upper.memory_counter:
                upper.learn()


        q_eval_value = upper.meta_eval_net.forward(torch.Tensor(state).cuda())[action]
        q_eval_list.append(q_eval_value)
        writer.add_scalar('scalar/q_eval', q_eval_value, i)

        r = copy.copy(reward)
        reward_list.append(r)
        writer.add_scalar('scalar/reward', r, i)

        collision_rate = collision_count / (i+1)
        collision_list.append(collision_rate)
        writer.add_scalar('scalar/collision_rate', collision_rate, i)

        if state[8] > state[3]:
            win_count += 1
        win_rate = win_count / (i+1)
        winner_list.append(win_rate)
        writer.add_scalar('scalar/win_rate', win_rate, i)

        # if (i + 1) % 250 == 0:
    ax[0].plot(reward_list[::], 'g-', label='total_loss')
    ax[1].plot(q_eval_list[::], 'b-', label="q_eval")
    ax[2].plot(collision_list[::], 'k-', label="collision_rate")
    ax[3].plot(winner_list[::], 'k-', label="win")
    plt.pause(0.001)
    
    output_name = "action_Vexp_ramp_acc_L0"
    # os.mkdir(output_path)
    for a in ax:
        a.legend()
    plt.savefig(os.path.join(output_path, output_name+".png"))
    writer.close()

    torch.save(obj=lower.eval_net.state_dict(), f=os.path.join(output_path,"eval.pth"))
    torch.save(obj=lower.target_net.state_dict(), f=os.path.join(output_path,"target.pth"))

    torch.save(obj=upper.meta_eval_net.state_dict(), f=os.path.join(output_path, "meta_eval.pth"))
    torch.save(obj=upper.meta_target_net.state_dict(), f=os.path.join(output_path, "meta_target.pth"))



if __name__ == '__main__':
    main()










# env = gym.make('merging_env-v0')
# for time_stamp in range(500):
#   if time_stamp < 200:
#     obs, rewards, done, info = env.step(1.1,1.2)
#   else:
#     obs, rewards, done, info = env.step(0,0)
    
#   env.render("human")

#   if done:
#     break

# for time_stamp in range(200):
#   obs, rewards, done, info = env.step(0,0)
#   env.render("human")

# env.reset()
