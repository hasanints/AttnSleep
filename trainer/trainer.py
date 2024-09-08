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
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = optimizer
        self.log_step = int(data_loader.batch_size)
        self.fold_id = fold_id
        self.class_weights = class_weights
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.selected = 0  # Initialize selected metrics

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

            if batch_idx == self.len_epoch:
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

            # Adjust learning rate
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001

        return train_log, overall_outs, overall_trgs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        """
        self.model.eval()
        self.valid_metrics.reset()
        outs = np.array([])
        trgs = np.array([])
        
        with torch.no_grad():
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

    def calculate_and_save_metrics(self, y_true, y_pred, stage_names):
        """
        Calculate and save metrics including confusion matrix, accuracy per stage, and overall metrics.
        """
        cm = confusion_matrix(y_true, y_pred, labels=range(len(stage_names)))
        cm_df = pd.DataFrame(cm, index=stage_names, columns=stage_names)
        cm_df.to_csv(f'{self.config["save_dir"]}/confusion_matrix_{self.fold_id}.csv', index=True)

        accuracy_per_stage = cm.diagonal() / cm.sum(axis=1)
        accuracy_per_stage_df = pd.DataFrame({
            'Stage': stage_names,
            'Accuracy': accuracy_per_stage
        })
        accuracy_per_stage_df.to_csv(f'{self.config["save_dir"]}/accuracy_per_stage_{self.fold_id}.csv', index=False)

        overall_accuracy = accuracy_score(y_true, y_pred)
        overall_f1 = f1_score(y_true, y_pred, average='macro')
        overall_kappa = cohen_kappa_score(y_true, y_pred)
        overall_mgm = np.mean(np.sqrt(accuracy_per_stage))

        overall_metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 Score', 'Kappa', 'MGm'],
            'Value': [overall_accuracy, overall_f1, overall_kappa, overall_mgm]
        })
        overall_metrics_df.to_csv(f'{self.config["save_dir"]}/overall_metrics_{self.fold_id}.csv', index=False)

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
