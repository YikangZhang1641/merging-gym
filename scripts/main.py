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

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.7
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

env = gym.make("merging_env-v0")
# env = gym.make("CartPole-v0")

env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 200)
        self.fc1.weight.data.uniform_(0,1)
        self.fc2 = nn.Linear(200,100)
        self.fc2.weight.data.uniform_(0,1)
        self.out = nn.Linear(100,NUM_ACTIONS)
        self.out.weight.data.uniform_(0,1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DRQN(nn.Module):
    def __init__(self, N_action):
        super(DRQN, self).__init__()
        self.lstm_i_dim = 16    # input dimension of LSTM
        self.lstm_h_dim = 16     # output dimension of LSTM
        self.lstm_N_layer = 1   # number of layers of LSTM
        self.N_action = N_action
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1)
        self.flat1 = Flatten()
        self.lstm = nn.LSTM(input_size=self.lstm_i_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)

        self.fc1 = nn.Linear(NUM_STATES, 200)
        self.fc1.weight.data.uniform_(0,1)
        self.fc2 = nn.Linear(200, self.lstm_i_dim)
        self.fc2.weight.data.uniform_(0,1)

        self.fc3 = nn.Linear(self.lstm_h_dim, 16)
        self.fc4 = nn.Linear(16, self.N_action)

    def forward(self, x, hidden):
        h1 = F.relu(self.fc1(x))
        h2 = self.fc2(h1)
        h3, new_hidden = self.lstm(h2, hidden)
        h4 = F.relu(self.fc3(h3))
        h5 = self.fc4(h4)
        return h5, new_hidden

class DQN():
    """docstring for DQN"""
    def __init__(self, load_path=None):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()
            
        if load_path is not None:
            self.eval_net.load_state_dict(torch.load(os.path.join(load_path, "eval.pth")))
            self.target_net.load_state_dict(torch.load(os.path.join(load_path, "target.pth")))


        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        # print("later state,", state)
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

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


def main():
    dqn = DQN()
    opponent = dqn
    # opponent = DQN("results/self")
    # opponent = DQN("L1")
    # opponent2 = DQN("L0")
    
    episodes = 4000
    print("Collecting Experience....")
    reward_list = []
    q_eval_list = []
    collision_list = []
    winner_list = []
    plt.ion()
    fig, ax = plt.subplots(4,1)
    ax[0].set_xlim(0, episodes)
    # ax[0].legend()
    ax[1].set_xlim(0, episodes)            
    # ax[1].legend()
    ax[2].set_xlim(0, episodes)         
    # ax[2].legend()
    ax[3].set_xlim(0, episodes)   
    # ax[3].legend()
    collision_count = 0
    win_count = 0

    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            env.render()

            action = dqn.choose_action(state)
            action_op = opponent.choose_action(state[3:] + state[:3])

            next_state, rewards, done, info = env.step(action, action_op)
            print("ego/op action,", action, action_op)

            if info["collision"]:
                collision_count += 1
                print("Collided!", collision_count / (i+1))

            reward, reward_op = rewards

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY and dqn.memory_counter:
                dqn.learn()
                # if done:
                #     print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                    
            if done:
                break
            state = next_state
        q_eval_value = dqn.eval_net.forward(torch.Tensor(state))[action]
        r = copy.copy(reward)
        reward_list.append(r)
        collision_list.append(collision_count / (i+1))
        if state[0] > state[3]:
            win_count += 1
        winner_list.append(win_count / (i+1))
        q_eval_list.append(q_eval_value)

        if (i + 1) % 500 == 0:
            ax[0].plot(reward_list, 'g-', label='total_loss')
            ax[1].plot(q_eval_list, 'b-', label="q_eval")
            ax[2].plot(collision_list, 'k-', label="collision_rate")
            ax[3].plot(winner_list, 'k-', label="win")
            plt.pause(0.001)
    
    output_path = datetime.datetime.now().strftime("%Y--%m--%d %H:%M:%S")
    output_name = "action_Vexp_ramp_acc_L0"
    os.mkdir(output_path)
    plt.savefig(os.path.join(output_path, output_name+".png"))

    torch.save(obj=dqn.eval_net.state_dict(), f=os.path.join(output_path,"eval.pth"))
    torch.save(obj=dqn.target_net.state_dict(), f=os.path.join(output_path,"target.pth"))



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
