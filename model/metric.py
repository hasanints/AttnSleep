import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score, f1_score
import pandas as pd
import numpy as np
import os
from os import walk



def g_mean(y_true, y_pred):
    """
    Menghitung Macro-averaged G-mean.
    """
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = np.diag(cm) / np.sum(cm, axis=1)
    gmean = np.prod(sensitivity)**(1.0/len(sensitivity))
    return gmean

def accuracy(output, target):
    """
    Calculate accuracy.
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def f1(output, target):
    """
    Calculate F1 score.
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1_score(pred.cpu().numpy(), target.data.cpu().numpy(), average='macro')


