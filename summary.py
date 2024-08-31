import argparse
import os
import re
import scipy.io as sio
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, f1_score

# Constants for sleep stages
W, N1, N2, N3, REM = 0, 1, 2, 3, 4
classes = ['W', 'N1', 'N2', 'N3', 'REM']
n_classes = len(classes)

def evaluate_metrics(cm):
    print("Confusion matrix:")
    print(cm)

    # Ensure the matrix is in float format for division
    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Handle division by zero
    TPR = np.divide(TP, TP + FN, out=np.zeros_like(TP), where=(TP + FN) != 0)
    TNR = np.divide(TN, TN + FP, out=np.zeros_like(TN), where=(TN + FP) != 0)
    PPV = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) != 0)
    NPV = np.divide(TN, TN + FN, out=np.zeros_like(TN), where=(TN + FN) != 0)
    FPR = np.divide(FP, FP + TN, out=np.zeros_like(FP), where=(FP + TN) != 0)
    FNR = np.divide(FN, TP + FN, out=np.zeros_like(FN), where=(TP + FN) != 0)
    FDR = np.divide(FP, TP + FP, out=np.zeros_like(FP), where=(TP + FP) != 0)

    # Overall accuracy and F1 Score
    ACC = np.divide((TP + TN), (TP + FP + FN + TN), out=np.zeros_like(TP), where=(TP + FP + FN + TN) != 0)
    ACC_macro = np.mean(ACC)
    F1 = np.divide(2 * PPV * TPR, (PPV + TPR), out=np.zeros_like(PPV), where=(PPV + TPR) != 0)
    F1_macro = np.mean(F1)

    print("Sample: {}".format(int(np.sum(cm))))
    for index_ in range(n_classes):
        print("{}: {}".format(classes[index_], int(TP[index_] + FN[index_])))

    return ACC_macro, ACC, F1_macro, F1, TPR, TNR, PPV

def print_performance(cm, y_true=[], y_pred=[]):
    tp = np.diagonal(cm).astype(float)
    tpfp = np.sum(cm, axis=0).astype(float)
    tpfn = np.sum(cm, axis=1).astype(float)
    acc = np.sum(tp) / np.sum(cm) if np.sum(cm) > 0 else 0  # Prevent division by zero
    precision = np.divide(tp, tpfp, out=np.zeros_like(tp), where=tpfp != 0)
    recall = np.divide(tp, tpfn, out=np.zeros_like(tp), where=tpfn != 0)
    f1 = np.divide(2 * precision * recall, (precision + recall), out=np.zeros_like(precision), where=(precision + recall) != 0)

    FP = cm.sum(axis=0).astype(float) - np.diag(cm)
    FN = cm.sum(axis=1).astype(float) - np.diag(cm)
    TP = np.diag(cm).astype(float)
    TN = cm.sum().astype(float) - (FP + FN + TP)
    specificity = np.divide(TN, TN + FP, out=np.zeros_like(TN), where=(TN + FP) != 0)
    mf1 = np.mean(f1)

    print("Sample: {}".format(np.sum(cm)))
    print("W: {}".format(tpfn[W]))
    print("N1: {}".format(tpfn[N1]))
    print("N2: {}".format(tpfn[N2]))
    print("N3: {}".format(tpfn[N3]))
    print("REM: {}".format(tpfn[REM]))
    print("Confusion matrix:")
    print(cm)
    print("Precision(PPV): {}".format(precision))
    print("Recall(Sensitivity): {}".format(recall))
    print("Specificity: {}".format(specificity))
    print("F1: {}".format(f1))
    if len(y_true) > 0:
        print("Overall accuracy: {}".format(np.mean(y_true == y_pred)))
        print("Cohen's kappa score: {}".format(cohen_kappa_score(y_true, y_pred)))
    else:
        print("Overall accuracy: {}".format(acc))
    print("Macro-F1 accuracy: {}".format(mf1))

def perf_overall(data_dir):
    # Remove non-output files, and perform ascending sort
    allfiles = os.listdir(data_dir)
    outputfiles = [os.path.join(data_dir, f) for f in allfiles if re.match("^output_.+\d+\.npz", f)]
    outputfiles.sort()

    y_true, y_pred = [], []
    for fpath in outputfiles:
        with np.load(fpath) as f:
            print(f["y_true"].shape)
            f_y_true = f["y_true"].flatten() if len(f["y_true"].shape) > 1 else np.hstack(f["y_true"])
            f_y_pred = f["y_pred"].flatten() if len(f["y_pred"].shape) > 1 else np.hstack(f["y_pred"])
            
            y_true.extend(f_y_true)
            y_pred.extend(f_y_pred)

            print("File: {}".format(fpath))
            cm = confusion_matrix(f_y_true, f_y_pred, labels=[0, 1, 2, 3, 4])
            print_performance(cm)

    print(" ")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sio.savemat('con_matrix_sleep.mat', {'y_true': y_true, 'y_pred': y_pred})
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    print("Ours:")
    print_performance(cm, y_true, y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="outputs_2013/outputs_eeg_fpz_cz",
                        help="Directory where to load prediction outputs")
    args = parser.parse_args()

    if args.data_dir is not None:
        perf_overall(data_dir=args.data_dir)

if __name__ == "__main__":
    main()
