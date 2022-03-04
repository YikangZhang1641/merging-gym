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
import math, random

import torch.optim as optim
import torch.autograd as autograd

from tensorboardX import SummaryWriter

# from common.layers import NoisyLinear
# from common.replay_buffer import ReplayBuffer

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.7
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

num_atoms = 51
Vmin = -10
Vmax = 10




# class Net(nn.Module):
#     """docstring for Net"""
#
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(NUM_STATES, 200)
#         self.fc1.weight.data.uniform_(0, 1)
#         self.fc2 = nn.Linear(200, 100)
#         self.fc2.weight.data.uniform_(0, 1)
#         self.out = nn.Linear(100, NUM_ACTIONS)
#         self.out.weight.data.uniform_(0, 1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         action_prob = self.out(x)
#         return action_prob
#
# class DQN():
#     """docstring for DQN"""
#
#     def __init__(self, load_path=None):
#         super(DQN, self).__init__()
#         self.eval_net, self.target_net = Net(), Net()
#
#         if load_path is not None:
#             self.eval_net.load_state_dict(torch.load(os.path.join(load_path, "eval.pth")))
#             self.target_net.load_state_dict(torch.load(os.path.join(load_path, "target.pth")))
#
#         self.learn_step_counter = 0
#         self.memory_counter = 0
#         self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
#         # why the NUM_STATE*2 +2
#         # When we store the memory, we put the state, action, reward and next_state in the memory
#         # here reward and action is a number, state is a ndarray
#         self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
#         self.loss_func = nn.MSELoss()
#
#     def choose_action(self, state):
#         state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array
#         # print("later state,", state)
#         if np.random.randn() <= EPISILO:  # greedy policy
#             action_value = self.eval_net.forward(state)
#             action = torch.max(action_value, 1)[1].data.numpy()
#             action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
#         else:  # random policy
#             action = np.random.randint(0, NUM_ACTIONS)
#             action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
#         return action
#
#     def store_transition(self, state, action, reward, next_state):
#         transition = np.hstack((state, [action, reward], next_state))
#         index = self.memory_counter % MEMORY_CAPACITY
#         self.memory[index, :] = transition
#         self.memory_counter += 1
#
#     def learn(self):
#
#         # update the parameters
#         if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
#             self.target_net.load_state_dict(self.eval_net.state_dict())
#         self.learn_step_counter += 1
#
#         # sample batch from memory
#         sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
#         batch_memory = self.memory[sample_index, :]
#         batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
#         batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
#         batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2])
#         batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])
#
#         # q_eval
#         q_eval = self.eval_net(batch_state).gather(1, batch_action)
#         q_next = self.target_net(batch_next_state).detach()
#         q_eval4next = self.eval_net(batch_next_state).detach()
#
#         # selected_q_next = q_next.max(1)[0]
#         max_act4next = q_eval4next.max(1)[1]
#         selected_q_next = q_next[range(128), max_act4next]
#
#         q_target = batch_reward + GAMMA * selected_q_next.view(BATCH_SIZE, 1)
#         loss = self.loss_func(q_eval, q_target)
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
import operator


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must for a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def push(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super(PrioritizedReplayBuffer, self).push(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, use_cuda, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.use_cuda = use_cuda
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.use_cuda:
            weight_epsilon = self.weight_epsilon.cuda()
            bias_epsilon = self.bias_epsilon.cuda()
        else:
            weight_epsilon = self.weight_epsilon
            bias_epsilon = self.bias_epsilon

        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class RainbowDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowDQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.linear1 = nn.Linear(num_inputs, 32)
        self.linear2 = nn.Linear(32, 64)

        self.noisy_value1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(64, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(64, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    def act(self, state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def projection_distribution(next_state, rewards, dones, target_model):
    batch_size = next_state.size(0)

    delta_z = float(Vmax - Vmin) / (num_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_atoms)

    next_dist = target_model(next_state).data.cpu() * support
    next_action = next_dist.sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist = next_dist.gather(1, next_action).squeeze(1)

    rewards = rewards.unsqueeze(1).expand_as(next_dist)
    dones = dones.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)

    Tz = rewards + (1 - dones) * 0.99 * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b = (Tz - Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long() \
        .unsqueeze(1).expand(batch_size, num_atoms)

    proj_dist = torch.zeros(next_dist.size())
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

    return proj_dist

def compute_td_loss(batch_size, replay_buffer, target_model, current_model, optimizer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action = Variable(torch.LongTensor(action))
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(np.float32(done))

    proj_dist = projection_distribution(next_state, reward, done, target_model)

    dist = current_model(state)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
    dist = dist.gather(1, action).squeeze(1)
    dist.data.clamp_(0.01, 0.99)
    loss = -(Variable(proj_dist) * dist.log()).sum(1)
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    current_model.reset_noise()
    target_model.reset_noise()

    return loss

def plot(frame_idx, rewards, losses):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.pause(1)
    # plt.show()

def main():
    load_path = "2022--03--03 14:42:31"
    output_path = datetime.datetime.now().strftime("%Y--%m--%d %H:%M:%S")
    writer = SummaryWriter(log_dir = output_path)

    env = gym.make("merging_env-v0")
    # env = env.unwrapped
    NUM_ACTIONS = env.action_space.n
    NUM_STATES = env.observation_space.shape[0]
    ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

    current_model = RainbowDQN(env.observation_space.shape[0], env.action_space.n, num_atoms, Vmin, Vmax)
    target_model = RainbowDQN(env.observation_space.shape[0], env.action_space.n, num_atoms, Vmin, Vmax)

    if load_path is not None:
        current_model.load_state_dict(torch.load(os.path.join(load_path, "eval.pth")))
        target_model.load_state_dict(torch.load(os.path.join(load_path, "target.pth")))

    if USE_CUDA:
        current_model = current_model.cuda()
        target_model = target_model.cuda()

    optimizer = optim.Adam(current_model.parameters(), 0.001)

    replay_buffer = ReplayBuffer(10000)
    update_target(current_model, target_model)

    # num_frames = 15000
    num_episodes = 4000
    batch_size = 32
    gamma = 0.99

    losses = []
    all_rewards = []
    episode_reward = 0

    state = env.reset()

    # for frame_idx in range(1, num_frames + 1):
    for epoc in range(num_episodes):
        episode_reward = 0
        state = env.reset()
        while True:
            # env.render()

            action = current_model.act(state)
            action_op = current_model.act(state[3:] + state[:3])
            # action_op = NUM_ACTIONS // 2
            next_state, [reward, reward_op], done, _ = env.step(action, action_op)

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                # all_rewards.append(episode_reward)
                break

            if len(replay_buffer) > batch_size:
                loss = compute_td_loss(batch_size, replay_buffer, target_model, current_model, optimizer)
                # losses.append(loss.data[0])
                # losses.append(loss.item())

        # if epoc % 5 == 4:
        #     plot(epoc, all_rewards, losses)

        if epoc % 20 == 19:
            update_target(current_model, target_model)

        writer.add_scalar('scalar/episode_reward', float(episode_reward), epoc)
        print("epoc", epoc, ", reward", episode_reward)
        # writer.add_scalar('scalar/loss', losses[-1], epoc)

    writer.close()

    torch.save(obj=current_model.state_dict(), f=os.path.join(output_path,"eval.pth"))
    torch.save(obj=target_model.state_dict(), f=os.path.join(output_path,"target.pth"))

if __name__ == '__main__':
    main()

