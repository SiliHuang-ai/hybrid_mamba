import numpy as np
import torch
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Trainer:

    def __init__(self, plan_to_go, h_model, plan_encoder, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.plan_to_go = plan_to_go
        # self.target_decoder_action = target_decoder_action
        self.h_model = h_model
        # self.plan_encoder = plan_encoder
        self.optimizer, self.plan_encoder_optimizer, self.l_optimizer = optimizer[0], optimizer[1], optimizer[2]
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler, self.plan_encoder_scheduler, self.l_scheduler = scheduler[0], scheduler[1], scheduler[2]
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()
        self.steps = []
        self.action_losses = []
        self.kl_losses = []
        self.returns = []
        self.lengths = []

    def train_iteration(self, results_dir, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.h_model.train()
        self.plan_to_go.train()
        # self.plan_encoder.train()
        # a = time.time()
        for _ in range(num_steps):
            action_loss, kl_loss = self.train_step()
            # if _==49:
            #     b = time.time()
            #     print("=" * 10)
            #     print("taining time: ", str((b - a)*200/3600))
            #     print("=" * 10)
            #     exit()
            self.action_losses.append(action_loss)
            self.kl_losses.append(kl_loss)
            self.steps.append(iter_num*num_steps+_)
            print("training steps:", str(iter_num*num_steps+_))
            if self.scheduler is not None:
                self.scheduler.step()
                # self.plan_encoder_scheduler.step()
                # self.l_scheduler.step()
            if _%50==0:
                _plt_training(self.steps, self.action_losses, self.kl_losses, results_dir, num=0)
            if _%500==0 and iter_num*num_steps+_>0:
                self.h_model.eval()
                self.plan_to_go.eval()
                # self.plan_encoder.eval()
                for eval_fn in self.eval_fns:
                    returns, length, target_rew = eval_fn(self.h_model, self.plan_to_go, 1)
                    _plt_testing(returns, 1, target_rew, results_dir, num=iter_num*num_steps+_)
                    # returns, length, target_rew = eval_fn(self.h_model, self.plan_to_go, self.plan_encoder, 2)
                    # _plt_testing(returns, 2, target_rew, results_dir, num=iter_num * num_steps + _)
                self.h_model.train()
                self.plan_to_go.train()
                # self.plan_encoder.train()
        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.h_model.eval()
        self.plan_to_go.eval()
        # self.plan_encoder.eval()
        # for eval_fn in self.eval_fns:
        #     returns, length, target_rew = eval_fn(self.h_model,self.l_model)
            # for k, v in outputs.items():
            #     logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(self.action_losses)
        logs['training/train_loss_std'] = np.std(self.action_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

def _plt_training(steps, action_loss, kl_loss, task_dir, num=0):
    plt.figure()
    # plt.axis([0, self.args.n_epoch, 0, 100])
    plt.cla()
    plt.plot(steps, action_loss)
    plt.xlabel('steps')
    plt.ylabel('action_loss')
    plt.savefig(task_dir + '/action_loss_{}.png'.format(num), format='png')
    # np.save(task_dir + '/action_loss_{}'.format(num), action_loss)
    # np.save(task_dir + '/steps_{}'.format(num), steps)

    plt.figure()
    plt.cla()
    plt.plot(steps, kl_loss)
    plt.xlabel('steps')
    plt.ylabel('kl_loss')
    plt.savefig(task_dir + '/kl_loss_{}.png'.format(num), format='png')
    # np.save(task_dir + '/kl_loss_{}'.format(num), kl_loss)


def _plt_testing(returns, rtg, target_rew, task_dir, num):
    plt.figure()
    # plt.axis([0, self.args.n_epoch, 0, 100])
    episodes = [i for i in range (len(returns))]
    plt.cla()
    plt.plot(episodes, returns)
    plt.xlabel('episodes')
    plt.ylabel('returns')
    plt.savefig(task_dir + '/'+ str(rtg)+'_'+str(target_rew)+'return_inference_{}.png'.format(num), format='png')
    np.save(task_dir + '/'+ str(rtg)+'_'+ str(target_rew) + 'return_inference_{}'.format(num), returns)
    # np.save(task_dir + '/steps_{}'.format(num), steps)

    # plt.figure()
    # plt.cla()
    # plt.plot(episodes, lengths)
    # plt.xlabel('steps')
    # plt.ylabel('lengths')
    # plt.savefig(task_dir + '/'+ str(target_rew)+'_lengths_''{}.png'.format(num), format='png')
    # np.save(task_dir + '/kl_loss_{}'.format(num), kl_loss)