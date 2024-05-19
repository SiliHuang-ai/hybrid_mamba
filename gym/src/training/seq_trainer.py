import numpy as np
import torch
from torch.autograd import Variable

from src.training.trainer import Trainer

def kl_divergence(mu1, logvar1, mu2, logvar2):
    total_kld = 0.5 * (logvar2 - logvar1 + (logvar1.exp() + (mu1 - mu2).pow(2)) / logvar2.exp() - 1)


    return total_kld

class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, h_s, h_g, h_rtg, h_r, h_timesteps, h_mask, h_loss_mask, h_d = self.get_batch(self.batch_size)

        action_target = torch.clone(actions)

        z_distribution_preds = self.h_model.forward(
            h_s, None,h_d, h_rtg[:, :], h_r,
        )

        h_action_preds = self.plan_to_go.get_predict_action(states, actions, z_distribution_preds, rewards, timesteps, None,attention_mask)
        act_dim = h_action_preds.shape[2]
        z_dim = h_g.shape[-1]
        z_target = torch.clone(h_g)
        z_mask = h_loss_mask*h_mask
        h_loss_mask = h_loss_mask.unsqueeze(-1).repeat([1, 1, attention_mask.shape[-1]]).reshape(-1,attention_mask.shape[-1])
        h_loss_mask = h_loss_mask * attention_mask

        h_action_preds = h_action_preds.reshape(-1, act_dim)[h_loss_mask.reshape(-1) > 0]
        h_action_target = action_target.reshape(-1, act_dim)[h_loss_mask.reshape(-1) > 0]
        h_loss = self.loss_fn(
            None, h_action_preds, None,
            None, h_action_target, None,
        )


        z_target = self.plan_to_go.map_g_to_z(h_g)
        l_action_preds = self.plan_to_go.get_predict_action(states, actions, None, rewards, timesteps,
                                                            z_target.detach(), attention_mask)
        l_action_preds = l_action_preds.reshape(-1, act_dim)[h_loss_mask.reshape(-1) > 0]
        z_loss = self.loss_fn(
            None, l_action_preds, None,
            None, h_action_target, None,
        )


        total_loss = h_loss + 0.5*z_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.plan_to_go.parameters(), .25)
        torch.nn.utils.clip_grad_norm_(self.h_model.parameters(), .25)
        self.optimizer.step()

        return total_loss.detach().cpu().item(), h_loss.detach().cpu().item()

    def update_target_networks(self,tau=1.0):
        for target_param, param in zip(self.target_decoder_action.parameters(), self.decoder_action.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )


    def update_networks(self,tau=1.0):
        for target_param, param in zip(self.decoder_action.parameters(), self.target_decoder_action.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def reparametrize(self,mu,logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps