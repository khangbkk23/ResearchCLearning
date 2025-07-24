# Torch environment
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.models import resnet34 as torchvision_resnet34
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

print(torch.__version__)
# ==== 1. Replay Buffer with per-class exemplar cap ====  
class ReplayBuffer():
    def __init__(self, max_per_class=1000):
        self.max_per_class = max_per_class # maximum exemplars per each class
        self.buffer = defaultdict(list) # buffer as a dict for each class
    
    def add_examples(self, x_batch, y_batch):
        for x, y in zip(x_batch, y_batch):
            cls = int(y.item())
            lst = self.buffer[cls]
            lst.append(x.detach().cpu())
            # FIFO replacement
            if len(lst) > self.max_per_class:
                lst.pop(0)
            self.buffer[cls] = lst
            
    def get_all_data(self):
        xs, ys = [], []
        for cls, examples in self.buffer.items():
            xs.append(torch.stack(examples))
            ys.append(torch.full((len(examples), ), cls, dtype=torch.long))
        
        if not xs:
            return None, None
        return torch.cat(xs, dim=0),torch.cat(ys, dim=0)
    
    # ==== 1. iCaRL buffer with memory budget memory_size ====
class iCaRLBuffer:
    def __init__(self, memory_size):
        self.memory_size = memory_size # maximum buffer size
        self.exemplar_sets = defaultdict(list) # class_id -> list of exemplar tensors
        self.seen_classes = set()

    def construct_exemplar_set(self, class_id, features, images, m):
        features = F.normalize(features, dim=1)  # important for cosine-based similarity
        class_mean = features.mean(0)
        # unsqueeze(0) to create a new dimension at position 0
        class_mean = F.normalize(class_mean.unsqueeze(0), dim=1)  # shape: [1, D]
        
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