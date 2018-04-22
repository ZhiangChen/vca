
"""
batch.py
Zhiang Chen
4/21/2018
"""
import random
import torch
import pickle

class Batch(object):
    def __init__(self, batch):
        E = torch.cat([exp[0][0] for exp in batch], 0)
        S = torch.cat([exp[0][1] for exp in batch], 0)
        G = torch.cat([exp[0][2] for exp in batch], 0)
        self.state = (E,S,G)

        E = torch.cat([exp[1][0] for exp in batch], 0)
        S = torch.cat([exp[1][1] for exp in batch], 0)
        self.next_state = (E, S, G)

        self.action = torch.cat([exp[0][3] for exp in batch], 0)

        self.reward = torch.Tensor([exp[2] for exp in batch]).unsqueeze(1)


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Batch(batch)

    def save(self, file="memory.p"):
        with open("./memory/"+file, 'wb') as wfp:
            pickle.dump(self.memory, wfp)
        print "saved memory: %d" % len(self.memory)

    def load(self, file="memory.p"):
        self.memory = pickle.load(open('./memory/'+file, 'rb'))
        self.position = len(self.memory)
        print "loaded memory: %d" % len(self.memory)


    def __len__(self):
        return len(self.memory)

