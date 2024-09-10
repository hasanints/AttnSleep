import torch
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score, f1_score
import pandas as pd
import numpy as np
import os
from os import walk

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

# class MetricsCalculator:
#     def __init__(self, config, checkpoint_dir):
#         self.config = config
#         self.checkpoint_dir = checkpoint_dir

#     def _calc_metrics(self):
#         n_folds = self.config["data_loader"]["args"]["num_folds"]
#         all_outs = []
#         all_trgs = []

#         outs_list = []
#         trgs_list = []
#         save_dir = os.path.abspath(os.path.join(self.checkpoint_dir, os.pardir))

#         # Collect output and target files
#         for root, dirs, files in os.walk(save_dir):
#             for file in files:
#                 if "outs" in file:
#                     outs_list.append(os.path.join(root, file))
#                 if "trgs" in file:
#                     trgs_list.append(os.path.join(root, file))

#         # If all folds have been processed
#         if len(outs_list) == n_folds:
#             for i in range(len(outs_list)):
#                 outs = np.load(outs_list[i])
#                 trgs = np.load(trgs_list[i])
#                 all_outs.extend(outs)
#                 all_trgs.extend(trgs)

#         all_trgs = np.array(all_trgs).astype(int)
#         all_outs = np.array(all_outs).astype(int)

#         # Convert to torch tensors for compatibility with defined accuracy and f1
#         all_outs_tensor = torch.tensor(all_outs)
#         all_trgs_tensor = torch.tensor(all_trgs)

#         # Calculate metrics using sklearn
#         r = classification_report(all_trgs, all_outs, digits=6, output_dict=True)
#         cm = confusion_matrix(all_trgs, all_outs)
#         df = pd.DataFrame(r)
#         df["cohen"] = cohen_kappa_score(all_trgs, all_outs)
#         df["accuracy"] = accuracy_score(all_trgs, all_outs) * 100  # Convert to percentage
#         df["f1_macro"] = f1_score(all_trgs, all_outs, average='macro') * 100  # Convert to percentage

#         # Calculate accuracy and F1 using custom functions
#         acc_custom = accuracy(all_outs_tensor, all_trgs_tensor)
#         f1_custom = f1(all_outs_tensor, all_trgs_tensor)
#         df["custom_accuracy"] = acc_custom * 100  # Convert to percentage
#         df["custom_f1_macro"] = f1_custom * 100  # Convert to percentage

#         # Save classification report to Excel
#         file_name = self.config["name"] + "_classification_report.xlsx"
#         report_Save_path = os.path.join(save_dir, file_name)
#         df.to_excel(report_Save_path)

#         # Save confusion matrix to file
#         cm_file_name = self.config["name"] + "_confusion_matrix.torch"
#         cm_Save_path = os.path.join(save_dir, cm_file_name)
#         torch.save(cm, cm_Save_path)
