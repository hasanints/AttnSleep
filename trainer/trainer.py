import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
import matplotlib.pyplot as plt

selected_d = {"outs": [], "trg": []}
class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, class_weights=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        # Initialize lists to store metrics
        self.train_loss = []
        self.valid_loss = []
        self.train_f1 = []
        self.valid_f1 = []
        
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer
        self.log_step = int(data_loader.batch_size) * 1  # reduce this if you want more logs

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

        self.fold_id = fold_id
        self.selected = 0
        self.class_weights = class_weights

    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
               total_epochs: Integer, the total number of epoch
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target, self.class_weights, self.device)

            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        self.train_loss.append(log['loss'])
        self.train_f1.append(log['f1_score'])  # Assuming f1_score is calculated and returned in log

        if self.do_validation:
            al_log, outs, trgs = self._valid_epoch(epoch)
            self.valid_loss.append(val_log['loss'])
            self.valid_f1.append(val_log['f1_score'])
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

            # THIS part is to reduce the learning rate after 10 epochs to 1e-4
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001

        return log, overall_outs, overall_trgs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            outs = np.array([])
            trgs = np.array([])
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target, self.class_weights, self.device)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

                preds_ = output.data.max(1, keepdim=True)[1].cpu()

                outs = np.append(outs, preds_.cpu().numpy())
                trgs = np.append(trgs, target.data.cpu().numpy())


        return self.valid_metrics.result(), outs, trgs

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def plot_metrics(trainer):
        epochs = range(1, len(trainer.train_loss) + 1)
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(epochs, trainer.train_loss, 'bo-', label='Training Loss')
        plt.plot(epochs, trainer.valid_loss, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, trainer.train_f1, 'bo-', label='Training F1 Score')
        plt.plot(epochs, trainer.valid_f1, 'ro-', label='Validation F1 Score')
        plt.title('Training and Validation F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()

        plt.tight_layout()
        plt.show()



