import copy
import numpy as np
from utils import soft_update

import torch

from model import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F

# CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING
# https://arxiv.org/pdf/1509.02971.pdf

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim, max_action).to(device) # Step 1 Pseudocode
        self.actor_target = copy.deepcopy(self.actor) # Step 2 Pseudocode
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device) # Step 1 Pseudocode
        self.critic_target = copy.deepcopy(self.critic)# Step 2 Pseudocode
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.discount = discount  # Step 1 Pseudocode
        self.tau = tau  # Step 1 Pseudocode

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64): #Step 3.vi.a Pseudocode
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size) #Step 3.vi Pseudocode

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state)) #Step 3.vi.a.b Pseudocode
        target_Q = reward + (not_done * self.discount * target_Q).detach() #Step 3.vi.a.b Pseudocode

        #Get current Q estimate
        current_Q = self.critic(state, action) #Step 3.vi.a.c Pseudocode

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q) #Step 3.vi.a.c Pseudocode

        # Optimize the critic
        self.critic_optimizer.zero_grad() #Step 3.vi.a.c Pseudocode
        critic_loss.backward() #Step 3.vi.a.c Pseudocode
        self.critic_optimizer.step() #Step 3.vi.a.c Pseudocode

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean() #Step 3.vi.a.d Pseudocode

        #Optimize the actor
        self.actor_optimizer.zero_grad() #Step 3.vi.a.d Pseudocode
        actor_loss.backward() #Step 3.vi.a.d Pseudocode
        self.actor_optimizer.step() #Step 3.vi.a.d Pseudocode

        # Update the frozen target models
        #Save the actor and critic and their optimizers
        #Update the frozen target models
        soft_update(self.critic_target, self.critic, self.tau) #Step 3.vi.a.e Pseudocode 
        soft_update(self.actor_target, self.actor, self.tau) #Step 3.vi.a.e Pseudocode


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    #Save the actor and critic and their optimizers
    def load(self, filename, evaluate=True):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        if evaluate:
            self.critic.eval()
            self.critic_target.eval()
            self.actor.eval()
            self.actor_target.eval()

        else:
            self.critic.train()
            self.critic_target.train()
            self.actor.train()
            self.actor_target.train()

