import torch
from abc import abstractmethod
from numpy import inf
import numpy as np
import matplotlib.pyplot as plt  # For visualizations
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # For confusion matrix visualization
import seaborn as sns  # For enhanced visualization (heatmaps)
from pathlib import Path  # To handle file paths correctly

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
    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        all_outs = []
        all_trgs = []
        metrics_log = {'loss': [], 'accuracy': []}  # Tracking loss and accuracy

        for epoch in range(self.start_epoch, self.epochs + 1):
            # Call _train_epoch method defined in Trainer class (inherited)
            result, epoch_outs, epoch_trgs = self._train_epoch(epoch, self.epochs)

            # Track metrics
            metrics_log['loss'].append(result.get('loss', 0))
            metrics_log['accuracy'].append(result.get('accuracy', 0))

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
        class_names = [str(i) for i in range(len(set(all_trgs)))]  # Example: class names as string of numbers
        self.plot_confusion_matrix(all_trgs, all_outs, class_names, save_path=self.checkpoint_dir / "confusion_matrix.png")
        
        # Pass metrics_log as a list of one dictionary
        self.plot_metrics([metrics_log], save_dir=self.checkpoint_dir)

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
        
        :param y_true: List of true labels
        :param y_pred: List of predicted labels
        :param class_names: List of class names for the confusion matrix display
        :param save_path: Path to save the confusion matrix image (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        
        if save_path:  # Save plot to file if save_path is provided
            plt.savefig(save_path)
            self.logger.info(f"Saved confusion matrix to {save_path}.")
        
        plt.show()  # Display the plot
        plt.close()  # Close the plot to free up memory

    def plot_metrics(self, metric_dicts, save_dir='.', metric_names=None):
        """
        Plots metrics like loss, accuracy over epochs for multiple runs and saves them as image files.

        :param metric_dicts: List of dictionaries containing lists of metrics over epochs 
                            (e.g., [{'loss': [...], 'accuracy': [...]}, {...}, ...])
        :param save_dir: Directory to save the metric plots (optional, defaults to current directory)
        :param metric_names: List of metric names to plot (optional, defaults to keys from metric_dict)
        """
        if not metric_names:
            # Assume all dicts have the same keys if metric_names is not provided
            metric_names = list(metric_dicts[0].keys())

        for metric_name in metric_names:
            plt.figure(figsize=(10, 6))
            
            # Plot each metric from each dictionary
            for i, metric_dict in enumerate(metric_dicts):
                values = metric_dict[metric_name]
                plt.plot(range(1, len(values) + 1), values, marker='o', label=f'Run {i+1} - {metric_name.capitalize()}')
            
            plt.title(f'Comparative {metric_name.capitalize()} over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name.capitalize())
            plt.legend()
            
            # Save plot to file if save_dir is provided
            save_path = f"{save_dir}/{metric_name}_comparative_over_epochs.png"
            plt.savefig(save_path)
            self.logger.info(f"Saved comparative {metric_name} plot to {save_path}.")
            
            plt.show()  # Display the plot
            plt.close()  # Close the plot to free up memory


    def _save_checkpoint(self, epoch, save_best=True):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
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

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _calc_metrics(self):
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

        if len(outs_list)==self.config["data_loader"]["args"]["num_folds"]:
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


        # Uncomment if you want to copy some of the important files into the experiement folder
        from shutil import copyfile
        copyfile("model/model.py", os.path.join(self.checkpoint_dir, "model.py"))
        copyfile("model/loss.py", os.path.join(self.checkpoint_dir, "loss.py"))
        copyfile("trainer/trainer.py", os.path.join(self.checkpoint_dir, "trainer.py"))
        copyfile("train_Kfold_CV.py", os.path.join(self.checkpoint_dir, "train_Kfold_CV.py"))
        copyfile("config.json",  os.path.join(self.checkpoint_dir, "config.json"))
        copyfile("data_loader/data_loaders.py",  os.path.join(self.checkpoint_dir, "data_loaders.py"))


