import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def load_and_visualize_confusion_matrix(confusion_matrix_path):
    """
    Load and visualize the confusion matrix stored as a PyTorch tensor.

    :param confusion_matrix_path: Path to the saved confusion matrix file.
    """
    # Load the confusion matrix
    cm = torch.load(confusion_matrix_path)

    # Convert the confusion matrix to a NumPy array if it is a PyTorch tensor
    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().numpy()

    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Example Usage
if __name__ == "__main__":
    save_dir = 'your/save/directory'  # Replace with your save directory
    confusion_matrix_file = os.path.join(save_dir, 'your_model_name_confusion_matrix.torch')  # Replace with your file name
    load_and_visualize_confusion_matrix(confusion_matrix_file)
