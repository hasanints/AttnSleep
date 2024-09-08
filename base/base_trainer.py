import torch
from abc import abstractmethod
from numpy import inf
import numpy as np
import matplotlib.pyplot as plt  # For visualizations
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, cohen_kappa_score
import seaborn as sns  # For enhanced visualization (heatmaps)
from pathlib import Path  # To handle file paths correctly
import pandas as pd

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, fold_id):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # Setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.fold_id = fold_id

        # Configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = Path(config.save_dir)  # Ensure this is a Path object

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        all_outs = []
        all_trgs = []
        metrics_log = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(self.start_epoch, self.epochs + 1):
            # Call _train_epoch method defined in Trainer class (inherited)
            result, epoch_outs, epoch_trgs = self._train_epoch(epoch, self.epochs)

            # Extract metrics
            train_loss, train_accuracy = result.get('loss', 0), result.get('accuracy', 0)
            val_loss, val_accuracy = result.get('val_loss', 0), result.get('val_accuracy', 0)

            # Track metrics
            metrics_log['train_loss'].append(train_loss)
            metrics_log['train_accuracy'].append(train_accuracy)
            metrics_log['val_loss'].append(val_loss)
            metrics_log['val_accuracy'].append(val_accuracy)

            # Save logged information into log dict
            log = {'epoch': epoch}
            log.update(result)
            all_outs.extend(epoch_outs)
            all_trgs.extend(epoch_trgs)

            # Print logged information to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # Evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # Check whether model performance improved or not, according to specified metric (mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(f"Metric '{self.mnt_metric}' is not found. Model performance monitoring is disabled.")
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

            if not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn’t improve for {} epochs. Training stops.".format(self.early_stop))
                break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        # After training is complete, plot confusion matrix and metrics
        class_names = [str(i) for i in range(len(set(all_trgs)))]
        self.plot_confusion_matrix(all_trgs, all_outs, class_names, save_path=self.checkpoint_dir / "confusion_matrix.png")
        self.plot_metrics(metrics_log, save_dir=self.checkpoint_dir)

        # Save final predictions and targets after training
        outs_name = "outs_" + str(self.fold_id)
        trgs_name = "trgs_" + str(self.fold_id)
        np.save(self.checkpoint_dir / outs_name, all_outs)
        np.save(self.checkpoint_dir / trgs_name, all_trgs)

        if self.fold_id == self.config["data_loader"]["args"]["num_folds"] - 1:
            self._calc_metrics()

    def _prepare_device(self, n_gpu_use):
        """
        Setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There's no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(f"Warning: The number of GPUs configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine.")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path=None):
        """
        Plot the confusion matrix and save it as an image file.
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')

        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved confusion matrix to {save_path}.")

        plt.show()
        plt.close()

    def plot_metrics(self, metrics_log, save_dir='.'):
        """
        Plots training and validation metrics like loss, accuracy over epochs and saves them as image files.
        """
        metric_pairs = [('train_loss', 'val_loss'), ('train_accuracy', 'val_accuracy')]

        for train_metric, val_metric in metric_pairs:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(metrics_log[train_metric]) + 1), metrics_log[train_metric], marker='o', label=f'Training {train_metric.split("_")[1].capitalize()}')
            plt.plot(range(1, len(metrics_log[val_metric]) + 1), metrics_log[val_metric], marker='o', label=f'Validation {val_metric.split("_")[1].capitalize()}')
            plt.title(f'Comparison of {train_metric.split("_")[1].capitalize()} over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(train_metric.split("_")[1].capitalize())
            plt.legend()
            save_path = f"{save_dir}/{train_metric.split('_')[1]}_comparison_over_epochs.png"
            plt.savefig(save_path)
            self.logger.info(f"Saved comparison plot for {train_metric.split('_')[1]} to {save_path}.")
            plt.show()
            plt.close()

    def _save_checkpoint(self, epoch, save_best=True):
        """
        Saving checkpoints
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # Load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # Load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _calc_metrics(self):
        """
        Calculate and save overall metrics like classification report and confusion matrix for all folds.
        """
        from sklearn.metrics import classification_report
        from sklearn.metrics import cohen_kappa_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        import pandas as pd
        import os
        from os import walk

        n_folds = self.config["data_loader"]["args"]["num_folds"]
        all_outs = []
        all_trgs = []

        outs_list = []
        trgs_list = []
        save_dir = os.path.abspath(os.path.join(self.checkpoint_dir, os.pardir))
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                if "outs" in file:
                    outs_list.append(os.path.join(root, file))
                if "trgs" in file:
                    trgs_list.append(os.path.join(root, file))

        if len(outs_list) == self.config["data_loader"]["args"]["num_folds"]:
            for i in range(len(outs_list)):
                outs = np.load(outs_list[i])
                trgs = np.load(trgs_list[i])
                all_outs.extend(outs)
                all_trgs.extend(trgs)

        all_trgs = np.array(all_trgs).astype(int)
        all_outs = np.array(all_outs).astype(int)

        r = classification_report(all_trgs, all_outs, digits=6, output_dict=True)
        cm = confusion_matrix(all_trgs, all_outs)
        df = pd.DataFrame(r)
        df["cohen"] = cohen_kappa_score(all_trgs, all_outs)
        df["accuracy"] = accuracy_score(all_trgs, all_outs)
        df = df * 100
        file_name = self.config["name"] + "_classification_report.xlsx"
        report_Save_path = os.path.join(save_dir, file_name)
        df.to_excel(report_Save_path)

        cm_file_name = self.config["name"] + "_confusion_matrix.torch"
        cm_Save_path = os.path.join(save_dir, cm_file_name)
        torch.save(cm, cm_Save_path)

        # Uncomment if you want to copy some of the important files into the experiment folder
        from shutil import copyfile
        copyfile("model/model.py", os.path.join(self.checkpoint_dir, "model.py"))
        copyfile("model/loss.py", os.path.join(self.checkpoint_dir, "loss.py"))
        copyfile("trainer/trainer.py", os.path.join(self.checkpoint_dir, "trainer.py"))
        copyfile("train_Kfold_CV.py", os.path.join(self.checkpoint_dir, "train_Kfold_CV.py"))
        copyfile("config.json", os.path.join(self.checkpoint_dir, "config.json"))
        copyfile("data_loader/data_loaders.py", os.path.join(self.checkpoint_dir, "data_loaders.py"))

