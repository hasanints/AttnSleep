import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import torch.nn as nn
import matplotlib.pyplot as plt  # Tambahkan import matplotlib

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

        # Tambahkan variabel untuk menyimpan riwayat loss
        self.train_loss_history = []
        self.valid_loss_history = [] if self.do_validation else None

    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :param total_epochs: Integer, the total number of epoch
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

        # Tambahkan riwayat loss rata-rata per epoch untuk visualisasi
        epoch_train_loss = self.train_metrics.avg('loss')
        self.train_loss_history.append(epoch_train_loss)

        # Jika validasi diaktifkan, lakukan validasi dan simpan loss-nya
        if self.do_validation:
            epoch_valid_loss = self._valid_epoch()  # Pastikan metode ini mengembalikan rata-rata validasi loss
            self.valid_loss_history.append(epoch_valid_loss)

        return self.train_metrics.result()

    def _valid_epoch(self):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target, self.class_weights, self.device)
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        # Mengembalikan loss rata-rata validasi untuk visualisasi
        return self.valid_metrics.avg('loss')

    def plot_loss(self):
        """Plot training and validation loss over epochs."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_history, label='Training Loss')
        if self.do_validation:
            plt.plot(self.valid_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.show()
