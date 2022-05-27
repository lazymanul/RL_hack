import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import numpy as np

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 2
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(input_shape, 64)
        self.lstm  = nn.LSTM(64,32)
        self.fc_pi = nn.Linear(32,num_actions)
        self.fc_v  = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()


from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.animation import AnimationMonitor

from pogema import GridConfig


def gen_actions(states, model, h_ins):
    actions = []
    h_outs = []
    for i in range(len(states)):
        s_current = states[i].flatten()
        prob, h_out = model.pi(torch.from_numpy(s_current).float(), h_ins[i]) 
        m = Categorical(prob)
        a = m.sample().item()
        actions.append(a)
        h_outs.append(h_out)
    return actions, h_outs

# Идея: шаг в направлении цели на радаре получает бонус
def navigation_reward(state, action):
    #import pdb
    #pdb.set_trace()
    moves = np.array([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)])
    center = np.array((5, 5))
    ind = np.unravel_index(np.argmax(state[2], axis=None), state[2].shape)
    dist1 = np.linalg.norm(ind - center)
    dist2 = np.linalg.norm(ind - (center + moves[action]))
    #print(f'dist1: {dist1}  dist2: {dist2}')
    if dist2 <= dist1:
        return 1 / 256
    else:
        return 0

# Идея: дополнительнй штраф за неподвижность
def stop_reward(s_current, s_next):
    if np.allclose(s_current, s_next):
        return  -2 / 256
    else:
        return 0

def store_data(states, actions, states_next, rewards, h_ins, h_outs, dones, model):    
    for i in range(len(states)):
        if dones[i]: 
            continue
        s_current = states[i].flatten()
        prob, _ = model.pi(torch.from_numpy(s_current).float(), h_ins[i])
        prob = prob.view(-1)
        s_next = states_next[i].flatten()
        a = actions[i]
        r = rewards[i] - 1 / 256
        #r += stop_reward(s_current, s_next) + navigation_reward(states[i], a)        
        # import pdb
        # pdb.set_trace()
        model.put_data((s_current, a, r, s_next, prob[a].item(), h_ins[i], h_outs[i], dones[i]))

if __name__ == "__main__":

    # Define random configuration
    grid_config = GridConfig(num_agents=1, # количество агентов на карте
                             size=8,      # размеры карты
                             density=0.3,  # плотность препятствий
                             seed=1,       # сид генерации задания 
                             max_episode_steps=256,  # максимальная длина эпизода
                             obs_radius=5, # радиус обзора
                            )

    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)

    # обновляем окружение
    obs = env.reset()

    model = PPO(obs[0].shape[0] * obs[0].shape[1] * obs[0].shape[2], env.action_space.n)


    score = 0.0
    print_interval = 20
    iterations = 200
    min_play_reward = 50

    def play_game():
        done = False
        states_current = env.reset()    
        while(not done): 
            s_current = states_current[0].flatten()       
            prob = model.pi(torch.from_numpy(s_current).float())
            m = Categorical(prob)
            a = m.sample().item()
            actions = [a]
            states_next, rewards, dones, info = env.step(actions)
            #env.render()            
            states_current = states_next

    for iteration in range(iterations):
        states_current = env.reset()
        h_outs = [(torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))] * 3
        dones = [False, ...]
        while not all(dones):
            for t in range(T_horizon): 
                h_ins = h_outs
                actions, h_outs = gen_actions(states_current, model, h_ins)
                states_next, rewards, dones, info = env.step(actions)
                store_data(states_current, actions, states_next, rewards, h_ins, h_outs, dones, model)
                states_current = states_next
                score += sum(rewards)

                if all(dones):
                    if score/print_interval > min_play_reward:
                        play_game()
                    break

            model.train_net()

        if iteration % print_interval == 0 and iteration != 0:
            print("# of episode :{}, avg score : {:.1f}".format(iteration, score/print_interval))
            score = 0.0

    env.close()

