import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn

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
            total_epochs: Integer, the total number of epochs
        :return: A log that contains average loss and metric in this epoch.
        """
        # Set the model to training mode
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []

        # Training loop
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)

            # Compute the loss
            loss = self.criterion(output, target, self.class_weights, self.device)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Update training metrics
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            # Logging for every log step
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} '.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                ))

            # Early stop condition
            if batch_idx == self.len_epoch:
                break

        # Collect training metrics
        train_log = self.train_metrics.result()

        # If validation is enabled, perform validation
        if self.do_validation:
            val_log, outs, trgs = self._valid_epoch(epoch)  # Perform validation
            # Add validation metrics to log with 'val_' prefix
            train_log.update(**{'val_' + k: v for k, v in val_log.items()})
            
            # If the current validation accuracy is the highest, update selected metrics
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
            
            # At the last epoch, save selected outputs and targets
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

            # Reduce the learning rate after 10 epochs
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001

        # Return training and validation logs along with overall outputs and targets
        return train_log, overall_outs, overall_trgs


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