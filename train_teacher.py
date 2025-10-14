import torch
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import gym
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
import wandb
from env.custom_hopper import *
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default=10_000_000, type=int, help='Number of training episodes')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--name', required=True, type=str, help='Choose a name for the run')

    return parser.parse_args()

args = parse_args()
seeds = [100, 200, 300, 400, 500]

if __name__ == "__main__":
    for seed in seeds:
        wandb.init(
            project="PPD",
            name=f"{args.name}_train_teacher_seed{seed}",
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
        config = wandb.config
        
        env = DummyVecEnv([lambda: gym.make('CustomHopper-source-v0') for _ in range(18)])
        env.seed(seed)
        env = VecNormalize(env, norm_obs=False, norm_reward = False)
        
        
        print("env creato")

        policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        activation_fn=torch.nn.ReLU)
        
        teacher_model = PPO("MlpPolicy", 
                            env,
                            verbose=1,
                            policy_kwargs=policy_kwargs, 
                            n_steps=256, 
                            batch_size=512, 
                            n_epochs=4, 
                            learning_rate=3e-4, 
                            gamma=0.995, 
                            ent_coef=0.01, 
                            tensorboard_log=f"tensorboard/seed_{seed}/",
                            seed=seed)
        
        # Enviroment for evaluation
        eval_env = DummyVecEnv([lambda: gym.make('CustomHopper-source-v0') for _ in range(1)])
        eval_env.seed(seed)
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward = False)

        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=None,
            log_path=f"{args.name}/eval_logs_seed_{seed}/",
            eval_freq=10000,
            n_eval_episodes = 5,
            deterministic=True,
            render=False,
            verbose=1
        )

        # wandb callback
        wandb_callback = WandbCallback(
            model_save_path=f"{args.name}/models/seed_{seed}/",
            verbose=2 )
                
        teacher_model.learn(total_timesteps=args.episodes, 
                            tb_log_name=f'{args.name}_teacher_seed{seed}',
                            callback=[wandb_callback, eval_callback])
        

        teacher_model.save(f'models/{args.name}_teacher_model_seed{seed}.ckpt')

        env = DummyVecEnv([lambda: gym.make('CustomHopper-source-v0') for _ in range(18)])
        env = VecNormalize(env, norm_obs=False)
        
        mean_reward, std_reward = evaluate_policy(teacher_model, teacher_model.get_env(), n_eval_episodes=10)
        print('Final teacher reward (raw): ', mean_reward, '+-', std_reward)

        mean_rewards = np.mean(eval_callback.evaluations_results, axis=1)
        np.savetxt(f"{args.name}/teacher_rewards_seed_{seed}.txt", mean_rewards, fmt='%.2f')

        wandb.finish()
