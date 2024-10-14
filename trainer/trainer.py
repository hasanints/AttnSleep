import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
import numpy as np

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
        """
        self.model.train()
        self.train_metrics.reset()
        overall_outs = []
        overall_trgs = []
        
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            # Panggil weighted_CrossEntropyLoss dengan class_weights dan device
            if self.class_weights is None:  # Jika tidak menggunakan class_weights
                loss = self.criterion(output, target)
            else:  # Jika menggunakan class_weights, pastikan kelas dan device diteruskan
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

        log = self.train_metrics.result()

        if self.do_validation:
            val_log, outs, trgs = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if val_log["accuracy"] > self.selected:
                self.selected = val_log["accuracy"]
                selected_d["outs"] = outs
                selected_d["trg"] = trgs
            if epoch == total_epochs:
                overall_outs.extend(selected_d["outs"])
                overall_trgs.extend(selected_d["trg"])

            # Kurangi learning rate setelah 10 epoch
            if epoch == 10:
                for g in self.lr_scheduler.param_groups:
                    g['lr'] = 0.0001

        return log, overall_outs, overall_trgs


    # def _valid_epoch(self, epoch):
    #     """
    #     Validate after training an epoch

    #     :param epoch: Integer, current training epoch.
    #     :return: A log that contains information about validation
    #     """
    #     self.model.eval()
    #     self.valid_metrics.reset()
    #     with torch.no_grad():
    #         outs = np.array([])
    #         trgs = np.array([])
    #         for batch_idx, (data, target) in enumerate(self.valid_data_loader):
    #             data, target = data.to(self.device), target.to(self.device)
    #             output = self.model(data)
                
    #             if self.class_weights is None:  # Gunakan CrossEntropyLoss standar (tanpa class weights)
    #                 loss = self.criterion(output, target)
    #             else:  # Gunakan class weights, tapi tanpa device sebagai argumen
    #                 loss = self.criterion(output, target, self.class_weights)

    #             self.valid_metrics.update('loss', loss.item())
    #             for met in self.metric_ftns:
    #                 self.valid_metrics.update(met.__name__, met(output, target))

    #             preds_ = output.data.max(1, keepdim=True)[1].cpu()

    #             outs = np.append(outs, preds_.cpu().numpy())
    #             trgs = np.append(trgs, target.data.cpu().numpy())


    #     return self.valid_metrics.result(), outs, trgs

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        """
        self.model.eval()
        self.valid_metrics.reset()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # Perhitungan loss
                loss = self.criterion(output, target)
                self.valid_metrics.update('loss', loss.item())
                preds = output.data.max(1, keepdim=True)[1]

                # Simpan prediksi dan true label untuk evaluasi
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(target.cpu().numpy())

        all_preds = np.array(all_preds).flatten()
        all_trues = np.array(all_trues).flatten()

        # Menghitung metrik evaluasi
        acc = accuracy_score(all_trues, all_preds)  # Akurasi
        mf1 = f1_score(all_trues, all_preds, average='macro')  # Macro F1-score
        kappa = cohen_kappa_score(all_trues, all_preds)  # Cohen's Kappa
        mgmean = g_mean(all_trues, all_preds)  # Macro G-mean

        # Precision, recall, F1-score per kelas
        precision_per_class = precision_score(all_trues, all_preds, average=None)
        recall_per_class = recall_score(all_trues, all_preds, average=None)
        f1_per_class = f1_score(all_trues, all_preds, average=None)

        # G-mean per kelas
        confusion_mtx = confusion_matrix(all_trues, all_preds)
        sensitivity_per_class = np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
        gmean_per_class = np.sqrt(sensitivity_per_class * (np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=0)))

        # Log hasil perhitungan
        self.logger.info(f'Accuracy: {acc}')
        self.logger.info(f'Macro F1-score: {mf1}')
        self.logger.info(f'Cohen Kappa: {kappa}')
        self.logger.info(f'Macro G-mean: {mgmean}')
        
        for i, (precision, recall, f1, gmean) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class, gmean_per_class)):
            self.logger.info(f'Class {i}: Precision={precision}, Recall={recall}, F1-score={f1}, G-mean={gmean}')
        
        # Mengembalikan semua metrik sebagai dictionary
        return {
            'accuracy': acc,
            'macro_f1': mf1,
            'cohen_kappa': kappa,
            'macro_gmean': mgmean,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'gmean_per_class': gmean_per_class
        }


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)