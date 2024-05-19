import numpy as np
import torch
import torch.nn as nn

import transformers

from src.models.model import TrajectoryModel
from src.models.trajectory_gpt2 import GPT2Model


class HDecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            z_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.z_dim=z_dim
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs
        )

        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size) #1000,128
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_reward = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)

        self.embed_done = nn.Embedding(3, hidden_size)
        self.embed_z = torch.nn.Linear(2*self.z_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)


        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_z_distribution = nn.Sequential(
            *([nn.Linear(hidden_size, 2*self.z_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, z_distributions, done, returns_to_go, h_r, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        z_embeddings = self.embed_z(z_distributions)
        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(h_r)
        time_embeddings = self.embed_timestep(timesteps)
        # done = done.reshape(done.shape[0],done.shape[1],1).to(dtype=torch.float32)
        done_embeddings = self.embed_done(done)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        z_embeddings = z_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        rewards_embeddings = rewards_embeddings + time_embeddings
        done_embeddings = done_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, z_embeddings, rewards_embeddings, done_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 5*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 5*seq_length)


        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        # state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        z_distribution_preds = self.predict_z_distribution(x[:,1])  # predict next action given state

        return z_distribution_preds

    def get_action(self, states, z_distributions, dones, returns_to_go, h_r, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        z_distributions = z_distributions.reshape(1, -1, 2*self.z_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        rewards = h_r.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        dones = dones.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            z_distributions = z_distributions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            rewards = rewards[:, -self.max_length:]
            timesteps = timesteps[:,-self.max_length:]
            dones = dones[:, -self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            z_distributions = torch.cat(
                [torch.zeros((z_distributions.shape[0], self.max_length - z_distributions.shape[1], 2*self.z_dim),
                             device=z_distributions.device), z_distributions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length - rewards.shape[1], 1),
                             device=rewards.device), rewards],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
            dones = torch.cat(
                [torch.ones((dones.shape[0], self.max_length - dones.shape[1]), device=timesteps.device) * 2,
                 dones],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        z_distributions_preds  = self.forward(
            states, z_distributions, dones, returns_to_go, rewards, timesteps, attention_mask=attention_mask, **kwargs)

        return z_distributions_preds[0,-1]
