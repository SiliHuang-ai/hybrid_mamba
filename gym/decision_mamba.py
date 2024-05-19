import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, fields, asdict
from mamba import Mamba, MambaConfig, RMSNorm

class MambaDMConfig(MambaConfig):


    def to_mamba_config(self) -> MambaConfig:
        mamba_config_fields = {field.name for field in fields(MambaConfig)}
        filtered_dict = {k: v for k, v in asdict(self).items() if k in mamba_config_fields}
        return MambaConfig(**filtered_dict)

class DecisionMamba(nn.Module):
    """
        This model uses Mamba to model (Return_1, state_1, action_1, Return_2, state_2, ...)
        """
    def __init__(
            self,
            # dm_config: MambaDMConfig,
            dm_config,
            state_dim,
            z_dim,
            hidden_size,
            action_tanh=True,
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.dm_config = dm_config
        self.config = dm_config.to_mamba_config()

        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)

        self.hidden_size = hidden_size

        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        # self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_reward = torch.nn.Linear(1, hidden_size)
        self.embed_done = nn.Embedding(3, hidden_size)
        self.embed_z = torch.nn.Linear(2 * self.z_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_z_distribution = nn.Sequential(
            *([nn.Linear(hidden_size, 2 * self.z_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_rewards = torch.nn.Linear(hidden_size, 1)
        self.predict_returns = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, z_distributions, done, returns_to_go, h_r):


        batch_size, seq_length = states.shape[0], states.shape[1]

        state_embeddings = self.embed_state(states)

        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(h_r)
        done_embeddings = self.embed_done(done)


        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, rewards_embeddings, done_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 4 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        x = self.mamba(stacked_inputs)
        x = self.norm_f(x)

        x = x.reshape(batch_size, seq_length, 4, self.hidden_size).permute(0, 2, 1, 3)


        z_distribution_preds = self.predict_z_distribution(x[:, 1])


        return z_distribution_preds

    def reset_eval(self,device):
        self.caches = [(None, torch.zeros(1, self.config.d_inner, self.config.d_conv - 1, device=device)) for _ in
                  range(self.config.n_layers)]

    def step(self, returns_to_go,states, z_distributions, rewards, done):

        if returns_to_go is not None:
            x = self.embed_return(returns_to_go)
        elif states is not None:
            x = self.embed_state(states)
        elif z_distributions is not None:
            x = self.embed_z(z_distributions)
        elif rewards is not None:
            x = self.embed_reward(rewards)
        else:
            x = self.embed_done(done)

        x, caches = self.mamba.step(x, self.caches)
        self.caches=caches
        x = self.norm_f(x)
        z_distribution_preds = None
        if states is not None:
            z_distribution_preds = self.predict_z_distribution(x)

        return z_distribution_preds

    def generate(self, tokenizer, prompt: str, num_tokens: int = 50, sample: bool = True, top_k: int = 40):
        self.eval()

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(
            next(self.parameters()).device)  #  (1, num_tokens)

        caches = [(None, torch.zeros(1, self.config.d_inner, self.config.d_conv - 1, device=input_ids.device)) for _ in
                  range(self.config.n_layers)]

        for i in range(input_ids.size(1) + num_tokens - 1):
            with torch.no_grad():

                next_token_logits, caches = self.step(input_ids[:, i], caches)

            if i + 1 >= input_ids.size(1):
                probs = F.softmax(next_token_logits, dim=-1)

                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k)
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)

                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token = torch.argmax(probs, dim=-1)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        output = [tokenizer.decode(output.tolist()) for output in input_ids][0]

        self.train()

        return output

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):


        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]


            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
