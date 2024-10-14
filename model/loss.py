import torch
import torch.nn as nn

# Weighted CrossEntropyLoss
def weighted_CrossEntropyLoss(output, target, classes_weights, device):
    # Pindahkan class_weights ke device (GPU atau CPU)
    weights = torch.tensor(classes_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    return criterion(output, target)

# Standard CrossEntropyLoss (tanpa class weights)
def CrossEntropyLoss(output, target):
    criterion = nn.CrossEntropyLoss()
    return criterion(output, target)

