import numpy as np
import torch
import torch.nn as nn
import transformers

from src.models.model import TrajectoryModel
from src.models.trajectory_gpt2 import GPT2Model
from torch.autograd import Variable


class DecoderAction(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            z_dim,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.z_dim=z_dim
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size) #1000,128
        self.embed_z = torch.nn.Linear(z_dim, hidden_size)
        # self.embed_z = torch.nn.Linear(2*z_dim, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_reward = torch.nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.embed_g_to_z = torch.nn.Linear(self.state_dim, z_dim)

        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )



    def forward(self, states, actions, z, rewards, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        z_embeddings = self.embed_z(z)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)


        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        rewards_embeddings = rewards_embeddings + time_embeddings
        z_embeddings = z_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (z_embeddings, state_embeddings, action_embeddings, rewards_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 4*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 4*seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)

        action_preds = self.predict_action(x[:,1])

        return action_preds

    def map_g_to_z(self, h_g):
        z = self.embed_g_to_z(h_g)
        return z

    def get_predict_action(self, states, actions, z_distributions, rewards, timesteps, goal,attention_mask=None):

        if goal is None:
            z_distributions = z_distributions.reshape(-1, 1, 2 * self.z_dim).repeat(1,states.size()[1],1)
            mu = z_distributions[:, :, :self.z_dim]
            logvar = z_distributions[:, :, self.z_dim:]
            z = self.reparametrize(mu, logvar)  # 64,2*z_dim
        else: z=goal.reshape(-1, 1, self.z_dim).repeat(1,states.size()[1],1)

        action_preds = self.forward(states, actions, z, rewards, timesteps, attention_mask)

        return action_preds

    def get_action(self,states, actions, z_distributions, rewards, timesteps):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        rewards = rewards.reshape(1, -1, 1)
        z_distributions = z_distributions.reshape(1, -1, 2*self.z_dim)
        mu = z_distributions[:, :, :self.z_dim]
        logvar = z_distributions[:, :, self.z_dim:]
        z = self.reparametrize(mu, logvar)

        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]
            z = z[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim),
                             device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length - rewards.shape[1], 1),
                             device=rewards.device), rewards],
                dim=1).to(dtype=torch.float32)

            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None
        action_preds = self.forward(states, actions, z, rewards, timesteps, attention_mask)
        return action_preds[0,-1]

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps

class DecoderAction1(nn.Module):

    def __init__(self, state_dim, act_dim, z_dim, hidden_size):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.z_dim = z_dim
        self.hidden_size = hidden_size

        layers = [nn.Linear(z_dim + self.state_dim, hidden_size)]
        for _ in range(1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.act_dim),
            nn.Tanh(),
        ])

        self.decoder_action = nn.Sequential(
            *layers
        )

    def forward(self, z_states):

        action_preds = self.decoder_action(z_states)
        return action_preds

    def get_primitive_action(self, states, z_distributions):
        mu = z_distributions[:, :self.z_dim]
        logvar = z_distributions[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)  # 64,z_dim
        z = z.unsqueeze(1).repeat([1, states.shape[1], 1])  # 64,20,z_dim
        z_states = torch.cat([z, states], -1)
        action_preds = self.forward(z_states)
        return action_preds

    def get_action(self, states, z_distributions):
        states = torch.tensor(states.reshape(1, self.state_dim), device=z_distributions.device).to(dtype=torch.float32)
        z_distributions = z_distributions.reshape(1, 2 * self.z_dim)
        mu = z_distributions[:, :self.z_dim]
        logvar = z_distributions[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)  # 64,2*z_dim

        z_states = torch.cat([z, states], -1)
        action_preds = self.forward(z_states)

        return action_preds[0]

    def get_predict_action(self, states, z_distributions):

        z_distributions = z_distributions.reshape(-1, 1, 2 * self.z_dim).repeat(1,states.size()[1],1)
        mu = z_distributions[:, :, :self.z_dim]
        logvar = z_distributions[:, :, self.z_dim:]
        z = self.reparametrize(mu, logvar)  # 64,2*z_dim

        z_states = torch.cat([z, states], -1)
        action_preds = self.forward(z_states)

        return action_preds

    def reparametrize(self,mu,logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps
