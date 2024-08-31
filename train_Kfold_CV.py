# train_Kfold_CV.py
import argparse
import collections
import os
import torch
import numpy as np
from parse_config import ConfigParser
from trainer.trainer import Trainer
from utils import prepare_device
from data_loader.data_loaders import load_folds_data, load_folds_data_shhs

# Import the new summary and visualize functions
from summary import evaluate_metrics
from visualize import plot_attention

def main(config, fold_id):
    # Setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data_loader, fold_id=fold_id)
    valid_data_loader = data_loader.split_validation()

    # Build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    print(model)

    # Prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # Build optimizer, learning rate scheduler.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    # Train the model
    trainer.train()

    # Perform visualization after training
    for i, (data, target) in enumerate(valid_data_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        plot_attention(output.detach().cpu().numpy())  # Visualize attention

    # Evaluate model performance and summarize
    outs, trgs = trainer.get_outputs_targets()  # Hypothetical function to get overall predictions and targets
    cm = confusion_matrix(trgs, outs)
    evaluate_metrics(cm)  # Print metrics summary

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str, help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str, help='Directory containing numpy files')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = []

    args2 = args.parse_args()
    fold_id = int(args2.fold_id)

    config = ConfigParser.from_args(args, fold_id, options)
    if "shhs" in args2.np_data_dir:
        folds_data = load_folds_data_shhs(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])
    else:
        folds_data = load_folds_data(args2.np_data_dir, config["data_loader"]["args"]["num_folds"])

    main(config, fold_id)
