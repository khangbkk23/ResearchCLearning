import torch
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

print(torch.__version__)
class CL():
    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        plt.xlabel('Episodes')

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)
        
# Main function
if __name__ == '__main__':
    pass