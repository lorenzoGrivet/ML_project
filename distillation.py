from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize,DummyVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import torch
from env.custom_hopper import *
from collections import deque
import wandb
import argparse
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

try:
    from sb3_distill.ppd import ProximalPolicyDistillation
except :
    import sys
    sys.path.append('../')
    from sb3_distill.ppd import ProximalPolicyDistillation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default=2_000_000, type=int, help='Number of distill episodes')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--name', required=True, type=str, help='Choose name same as teacher')
    parser.add_argument('--size', required=True, type=str, help='Choose size: small, same, large')
    return parser.parse_args()

args = parse_args()
seeds = [100, 200, 300, 400, 500]

def moving_avg(rewards, window=10):
    running_avg = []
    avg_deque = deque(maxlen=window)
    normed_rewards = []
    for r in rewards:
        avg_deque.append(r)
        avg = sum(avg_deque) / len(avg_deque)
        running_avg.append(avg)
        normed_rewards.append(r / (avg + 1e-8))  # normalize wrt the moving average
    mean_raw = np.mean(rewards)
    mean_norm = np.mean(normed_rewards)
    std_raw = np.std(rewards)
    std_norm = np.std(normed_rewards)
    return normed_rewards, running_avg, mean_raw, std_raw, mean_norm, std_norm

if __name__ == "__main__":
    for seed in seeds:
    
        wandb.init(
            project="PPD",
            name=f"{args.name}_distill_{args.size}_seed{seed}",
            entity="andrea-gaudino02-politecnico-di-torino",
            config={
                "env": "CustomHopper-source-v0",
                "algorithm": "PPD",
                "total_timesteps": args.episodes,
                "seed": seed
            },
            sync_tensorboard=True,
            monitor_gym=True
        )
        
        env = DummyVecEnv([lambda: gym.make('CustomHopper-source-v0') for _ in range(18)])
        env.seed(seed)
        env = VecNormalize(env, norm_obs=False, norm_reward=False)
        obs_space = env.observation_space
        act_space = env.action_space


        teacher_model = PPO.load(f'models/{args.name}_teacher_model_seed{seed}.ckpt', env=env,
                                 custom_objects={
                                        "observation_space": obs_space,
                                        "action_space": act_space
                                    })

        #model size
        if args.size == 'small':
            policy_kwargs = dict(
            net_arch=[dict(pi=[128, 128], vf=[64, 64])],
            activation_fn=torch.nn.ReLU)
        elif args.size == 'same':
            policy_kwargs = dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=torch.nn.ReLU)
        elif args.size == 'large':
            policy_kwargs = dict(
            net_arch=[dict(pi=[512, 512], vf=[512, 512])],
            activation_fn=torch.nn.ReLU)


        student_model = ProximalPolicyDistillation("MlpPolicy", 
                                                   env, 
                                                   verbose=1, 
                                                   policy_kwargs=policy_kwargs, 
                                                   n_steps=64, 
                                                   batch_size=512, 
                                                   n_epochs=4, 
                                                   learning_rate=3e-4, 
                                                   gamma=0.995, 
                                                   ent_coef=0, 
                                                   tensorboard_log=f"tensorboard/seed_{seed}/", 
                                                   seed=seed)

        # Constant distillation-loss weight
        student_model.set_teacher(teacher_model, distill_lambda=5)
        # Linearly annealing distillation-loss weight

        eval_env = DummyVecEnv([lambda: gym.make('CustomHopper-source-v0') for _ in range(1)])
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False)
        
        # Callback for evaluating the student
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=None,
            log_path=f"{args.name}/distill_{args.size}_eval_logs_seed_{seed}/",
            eval_freq=2000,
            n_eval_episodes = 5,
            deterministic=True,
            render=False,
            verbose=1
        )

        #wandb callback
        wandb_callback = WandbCallback(
            model_save_path=f"{args.name}/models/seed_{seed}/",
            verbose=2 )

        student_model.learn(total_timesteps=args.episodes,
                            tb_log_name=args.name+"_distill", 
                            callback=[wandb_callback, eval_callback])

       
        #student distilled model
        student_model.save(f'models/{args.name}_{args.size}_student_model_seed{seed}.ckpt')
        
        #evaluation
        rewards, _ = evaluate_policy(teacher_model, teacher_model.get_env(), n_eval_episodes=50, deterministic=True, return_episode_rewards=True)
        normed_rewards, running_avg, mean_raw, std_raw, mean_norm, std_norm=moving_avg(rewards)
        print('Raw teacher score: ', mean_raw, '+-', std_raw)
        print('Normalized teacher score: ', mean_norm, '+-', std_norm)
        
        
        rewards, _ = evaluate_policy(student_model, student_model.get_env(), n_eval_episodes=50, deterministic=True, return_episode_rewards=True)
        normed_rewards, running_avg, mean_raw, std_raw, mean_norm, std_norm=moving_avg(rewards)
        print('Raw student score: ', mean_raw, '+-', std_raw)
        print('Normalized student score: ', mean_norm, '+-', std_norm)

        mean_rewards = np.mean(eval_callback.evaluations_results, axis=1)
        np.savetxt(f"{args.name}/distill_{args.size}_rewards_seed_{seed}.txt", mean_rewards, fmt='%.2f', header='mean_reward')

        wandb.finish()