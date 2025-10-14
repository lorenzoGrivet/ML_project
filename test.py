"""Test an RL agent trained with Stable-Baselines3 on the CustomHopper environment"""
import argparse
import os
import time
import wandb
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200, help='Number of test episodes')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--name', type=str, required=True, help='Same name of the train')
    parser.add_argument('--model', type=str, required=True, help='teacher or distill')
    parser.add_argument('--size', type=str, default='same', help='Size of the model: small, same, large')
    return parser.parse_args()

seeds = [100, 200, 300, 400, 500]

def main(seed):
    args = parse_args()

    # Output file
    os.makedirs(args.name, exist_ok=True)
    if args.model == 'teacher':
        out_file_path = os.path.join(args.name, f"{args.name}_{args.model}_output_test_seed{seed}.txt")
    if args.model == 'distill':
        out_file_path = os.path.join(args.name, f"{args.name}_{args.model}_{args.size}_output_test_seed{seed}.txt")
    out_file = open(out_file_path, "w")

    # Enviroment initialization
    env = gym.make(f'CustomHopper-target-v0')
    env.seed(seed)
    
    # W&B initialization
    nameRun = 'ppd'
    if args.model == 'teacher':
        nameRun = f"{args.name}_{args.model}_test_seed{seed}"
    if args.model == 'distill':
        nameRun = f"{args.name}_{args.model}_{args.size}_test_seed{seed}"


    wandb.init(
        project="PPD",
        name=nameRun,
        entity="andrea-gaudino02-politecnico-di-torino",
        config={
            "episodes": args.episodes,
            "render": args.render,
            "env": "CustomHopper-target-v0"
        }
    )
    
    # Uploading SB3 model
    model_path = ''
    if args.model == 'teacher':
        model_path = f"models/{args.name}_teacher_model_seed{seed}.ckpt"
    if args.model == 'distill':
        model_path = f"models/{args.name}_{args.size}_student_model_seed{seed}.ckpt"

    model = PPO.load(model_path, env=env,
                     custom_objects={
                        "observation_space": env.observation_space,
                        "action_space": env.action_space
                    })


    for episode in range(1, args.episodes + 1):
        done = False
        state = env.reset()
        test_reward = 0
        start_time = time.time()

        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            test_reward += reward

        duration = time.time() - start_time

        print(f"Episode {episode} | Return: {test_reward:.2f} | Duration: {duration:.2f}s")
        out_file.write(f"Episode {episode} | Return: {test_reward:.2f} | Duration: {duration:.2f}s\n")
        
        wandb.log({
            "episode": episode,
            "test_reward": test_reward,
            "test_duration": duration
        }, step=episode)
        

    out_file.close()
    wandb.finish()

if __name__ == '__main__':
    for seed in seeds:
        main(seed)
