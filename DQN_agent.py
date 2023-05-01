import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.input_shape = input_shape
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.n_actions = n_actions

    def store_transition(self, state, action, reward, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]

        return states, actions, rewards, states_


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.A = nn.Linear(64, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        A = self.A(flat2)  # A is the action set
        return A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        #self.load_state_dict(T.load(self.checkpoint_file))
        self.load_state_dict(T.load(self.checkpoint_file,map_location=T.device('cpu')))

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min,
                 replace, chkpt_dir, q_eval_name, q_next_name, instance):
        # path='D:\PyCharmCommunityEdition2020.2.1\Projects\GoogleColab'
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_init = epsilon
        self.epsilon_min = eps_min
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.mem_size = mem_size
        self.memory = ReplayBuffer(mem_size, input_dims, self.n_actions)

        T.manual_seed(instance) # Fixed the initial random parameters
        #Local Network*
        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name=q_eval_name,
                                          chkpt_dir=self.chkpt_dir)
        # Target Network*
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                          input_dims=self.input_dims,
                                          name=q_next_name,
                                          chkpt_dir=self.chkpt_dir)

    def set_loadchkpt_dir(self,chkpt_dir, q_eval_name, q_next_name):
        self.q_eval.checkpoint_dir = chkpt_dir
        self.q_eval.checkpoint_file = os.path.join(chkpt_dir, q_eval_name)
        self.q_next.checkpoint_dir = chkpt_dir
        self.q_next.checkpoint_file = os.path.join(chkpt_dir, q_next_name)

    def set_savechkpt_dir(self,chkpt_dir, q_eval_name, q_next_name):
        self.q_eval.checkpoint_dir = chkpt_dir
        self.q_eval.checkpoint_file = os.path.join(chkpt_dir, q_eval_name)
        self.q_next.checkpoint_dir = chkpt_dir
        self.q_next.checkpoint_file = os.path.join(chkpt_dir, q_next_name)

    def initialization_FIFO(self, buffer):
        for idx in np.arange(0,len(buffer['obs'])):
            #self.store_transition(buffer['obs'][idx], np.array([buffer['action'][idx]]),
            #                 np.array([buffer['reward'][idx]]), buffer['next_obs'][idx])
            index = self.memory.mem_cntr % self.mem_size
            self.memory.state_memory[index] = buffer['obs'][idx]
            self.memory.new_state_memory[index] = buffer['next_obs'][idx]
            actions = np.zeros(self.n_actions)
            actions[np.array([buffer['action'][idx]])] = 1.0
            self.memory.action_memory[index] = actions
            self.memory.reward_memory[index] = np.array([buffer['reward'][idx]])
            self.memory.mem_cntr += 1

    def reset_buffer(self):
        self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)

    def choose_action(self, observation, random_epsilon):
        if random_epsilon >= self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
            return action
        else:
            action = np.random.choice(self.action_space)
        return action

    def choose_action_test(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
        advantage = self.q_eval.forward(state)
        action = T.argmax(advantage).item()
        return action

    def store_transition(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decay_epsfix(self, episodes, episode_step):
        decay_rate = max((episodes - episode_step)/episodes,0)
        self.epsilon = (self.epsilon_init - self.epsilon_min)* decay_rate + self.epsilon_min

    def epsilon_reset(self,epsilon_aux):
        self.epsilon = epsilon_aux

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr <= self.batch_size:
            return
        state, action, reward, new_state = \
            self.memory.sample_buffer(self.batch_size)


        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        action_values = np.array(self.action_space, dtype=np.int32)
        action_indices = np.dot(action, action_values)

        q_eval = self.q_eval.forward(states).to(self.q_eval.device)
        q_next = self.q_next.forward(states_).to(self.q_eval.device)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        'state_action_values'
        state_action_values = q_eval[batch_index, action_indices]
        'expected Q values'
        expected_state_action_values = rewards + self.gamma * T.max(q_next, dim=1)[0]


        TD = expected_state_action_values - state_action_values
        loss = TD.pow(2).to(self.q_eval.device)
        loss = loss.mean()

        # Optimize the model
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()


        self.replace_target_network()  # Update Network
        self.learn_step_counter += 1
