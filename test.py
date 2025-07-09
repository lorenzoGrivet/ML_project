"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse
import torch
import gym
from env.custom_hopper import *
from agent import Agent, Policy
import wandb
import os
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100, type=int, help='Number of test episodes')
    parser.add_argument('--name', default='hopper-test_noName', type=str, help='Same as training run name')
    return parser.parse_args()

args = parse_args()

def main():
    
    os.makedirs(args.name, exist_ok=True)
    out_file_name = f"{args.name}/output_test.txt"
    out_file = open(out_file_name, "w")

    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    wandb.init(
        project="Confronti_progetti",
        name=f"{args.name}_test",
        entity="andrea-gaudino02-politecnico-di-torino",
        config={
            "env": "CustomHopper-source-v0",
            "model_path": f"{args.name}/model.mdl",  
            "device": args.device,
            "episodes": args.episodes
        }
    )

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())
    print("Rendering: ", args.render)
    
    out_file.write(f"Action space: {env.action_space}\n")
    out_file.write(f"State space: {env.observation_space}\n")
    out_file.write(f"Dynamic parameters: {env.get_parameters()}\n")
    out_file.write(f"Rendering: {args.render}\n")

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(torch.load(f"{args.name}/model.mdl"), strict=True)

    agent = Agent(policy, device=args.device)

    for episode in range(args.episodes):

        start = time.time()

        done = False
        test_reward = 0
        state = env.reset()

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, info = env.step(action)

            if args.render:
                env.render()

            test_reward += reward

        end = time.time()

        tempo_test = end - start
        print(f"Episode: {episode+1} | Return: {test_reward}")
        out_file.write(f"Episode: {episode+1} | Return: {test_reward}\n")
        wandb.log({"episode": episode+1, "test_reward": test_reward, "tempo_test": tempo_test}, step=episode+1)

    out_file.close()
    wandb.finish()

if __name__ == '__main__':
    main()
