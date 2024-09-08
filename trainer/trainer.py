import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score

selected_d = {"outs": [], "trg": []}

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, fold_id,
                 valid_data_loader=None, class_weights=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, fold_id)
        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)  # Initialize len_epoch as the length of the data loader

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
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))

            if batch_idx == self.len_epoch:  # Use self.len_epoch here to determine when to stop
                break

        train_log = self.train_metrics.result()

        if self.do_validation:
            val_log, outs, trgs = self._valid_epoch(epoch)
            train_log.update(**{'val_' + k: v for k, v in val_log.items()})

            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs

            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001

        return train_log, overall_outs, overall_trgs

    def _progress(self, batch_idx):
        """
        Calculate training progress.
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch  # Use self.len_epoch here
        return base.format(current, total, 100.0 * current / total)
