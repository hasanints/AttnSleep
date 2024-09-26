import torch
import torch.nn as nn

# Weighted CrossEntropyLoss
def weighted_CrossEntropyLoss(output, target, classes_weights, device):
    cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
    return cr(output, target)

# Standard CrossEntropyLoss (No Class Weights)
def CrossEntropyLoss(output, target, device=None):
    cr = nn.CrossEntropyLoss()  # Standard CrossEntropyLoss without weights
    return cr(output, target)
