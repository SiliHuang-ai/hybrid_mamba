import numpy as np
import torch
import torch.nn as nn

import transformers

from src.models.model import TrajectoryModel
from src.models.trajectory_gpt2 import GPT2Model
from torch.autograd import Variable

class PrimitivePolicy(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            # decoder_action,
            # target_decoder_action,
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

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size) #1000,128
        # self.embed_rewards = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_z = torch.nn.Linear(hidden_size, 2*z_dim)


        #decoder action
        # self.decoder_action = decoder_action
        # self.target_decoder_action = target_decoder_action

        # layers = [nn.Linear(z_dim+self.state_dim, hidden_size)]
        # for _ in range(1):
        #     layers.extend([
        #         nn.ReLU(),
        #         nn.Dropout(0.1),
        #         nn.Linear(hidden_size, hidden_size)
        #     ])
        # layers.extend([
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_size, self.act_dim),
        #     nn.Tanh(),
        # ])
        #
        # self.decoder_action = nn.Sequential(
        #     *layers
        # )

    def forward(self, states, actions, rewards, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        # rewards_embeddings = self.embed_rewards(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        # rewards_embeddings = rewards_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        # # get avg predictions
        # z_distributions = self.predict_z(x[:,1])  # predict z given last state and action
        # z_mask = attention_mask.unsqueeze(-1).repeat([1, 1, 2*self.z_dim])
        # z_distributions = torch.sum(z_mask * z_distributions, axis=1) / torch.sum(z_mask, axis=1)
        # mu = z_distributions[:,:self.z_dim]
        # logvar = z_distributions[:,self.z_dim:]

        #get predictions
        z_distributions = self.predict_z(x[:, 1][:, -1, :])

        # action_preds = self.decoder_action.get_primitive_action(states, z_distributions)  # predict next action given state
        # z_distributions = torch.cat([mu, logvar], -1)
        # return action_preds, z_distributions
        return z_distributions

    def get_actual_distribution(self, states, actions, rewards, timesteps):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        # rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            # rewards = rewards[:, -self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            # rewards = torch.cat(
            #     [torch.zeros((rewards.shape[0], self.max_length - rewards.shape[1], 1),
            #                  device=rewards.device), rewards],
            #     dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        z_distributions = self.forward(
            states, actions, None, timesteps, attention_mask=attention_mask)

        return z_distributions[-1]


    def get_action(self, states, actions, z_distributions, timesteps):
        # we don't care about the past rewards in this model
        # states = torch.tensor(states.reshape(1, self.state_dim), device=z_distributions.device).to(dtype=torch.float32)
        # z_distributions = z_distributions.reshape(1, 2*self.z_dim)
        # mu = z_distributions[:, :self.z_dim]
        # logvar = z_distributions[:, self.z_dim:]
        # z = self.reparametrize(mu, logvar)  # 64,2*z_dim
        # z_states = torch.cat([z, states], -1)
        # action_preds = self.decoder_action(z_states)
        # return action_preds[0]

        action_preds = self.decoder_action.get_action(states, actions, z_distributions, timesteps)
        return action_preds

    def get_predict_action(self, states, z_distributions):
        # 扩展z_distribution与state保持一致
        # z_distributions = z_distributions.reshape(-1, 1, 2 * self.z_dim).repeat(1,states.size()[1],1)
        # mu = z_distributions[:, :, :self.z_dim]
        # logvar = z_distributions[:, :, self.z_dim:]
        # z = self.reparametrize(mu, logvar)  # 64,2*z_dim
        # z_states = torch.cat([z, states], -1)
        # action_preds = self.decoder_action(z_states)
        action_preds = self.decoder_action.get_predict_action(states, z_distributions)
        return action_preds

    def reparametrize(self,mu,logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps
