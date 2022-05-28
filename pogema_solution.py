import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_shape=363
        self.num_actions=5
        self.load_weights=False

        self.learning_rate = 0.0005
        self.gamma         = 0.98
        self.lmbda         = 0.95
        self.eps_clip      = 0.1
        self.K_epoch       = 3
        self.T_horizon     = 120

        self.data = []
             
        self.fc1   = nn.Linear(self.input_shape, 256)
        self.fc_pi = nn.Linear(256, self.num_actions)
        self.fc_v  = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.load_weights:
            self.load_state_dict(torch.load("pogema_ppo_weights.h5"))
            self.eval()


    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        if s.shape == (1, 0):
            return

        for i in range(self.K_epoch):            
            td_target = r + self.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def act(self, obs, dones=None, positions_xy=None, targets_xy=None) -> list:
        """
        Given current observations, Done flags, agents' current positions and their targets, produce actions for agents.
        """
        actions = []
        for state in obs:
            s_current = state.flatten()
            prob = self.pi(torch.from_numpy(s_current).float())            
            m = Categorical(prob)
            a = m.sample().item()
            actions.append(a)
        return actions

def store_data(states, actions, states_next, rewards, dones, model):    
    for i in range(len(states)):        
        s_current = states[i].flatten()
        prob = model.pi(torch.from_numpy(s_current).float())        
        s_next = states_next[i].flatten()
        a = actions[i]
        r = rewards[i] - 1 / 256
        #r += stop_reward(s_current, s_next) + navigation_reward(states[i], a)        
        model.put_data((s_current, a, r, s_next, prob[a].item(), dones[i]))

def play_game():    
        states_current = env.reset()    
        dones = [False, ...]
        while(not all(dones)): 
            actions = model.act(states_current)
            states_next, rewards, dones, info = env.step(actions)
            #env.render()            
            states_current = states_next

from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.animation import AnimationMonitor
from pogema import GridConfig

if __name__ == "__main__":

    # Define random configuration
    grid_config = GridConfig(num_agents=1, # количество агентов на карте
                             size=20,      # размеры карты
                             density=0.3,  # плотность препятствий
                             seed=1,       # сид генерации задания 
                             max_episode_steps=256,  # максимальная длина эпизода
                             obs_radius=5, # радиус обзора
                            )

    env = gym.make("Pogema-v0", grid_config=grid_config)
    env = AnimationMonitor(env)

    # обновляем окружение
    obs = env.reset()

    model = Model()

    score = 0.0
    print_interval = 30
    iterations = 1000
    min_play_reward = 10

    for iteration in range(iterations):
        states_current = env.reset()
        dones = [False, ...]
        while not all(dones):
            for t in range(model.T_horizon):
                actions = model.act(states_current)
                states_next, rewards, dones, info = env.step(actions)
                store_data(states_current, actions, states_next, rewards, dones, model)
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
    torch.save(model.state_dict(), "pogema_ppo_weights.h5")