Train.py

import os
import sys
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from dataset import *

class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def test_model(model, test_dataloader, cls_criterion, device, epoch):
    print("")
    model.eval()
    test_metrics = {"loss": [], "acc": []}
    for batch_i, (X, y) in enumerate(test_dataloader):
        image_sequences = Variable(X.to(device), requires_grad=False)
        labels = Variable(y, requires_grad=False).to(device)
        with torch.no_grad():
            model.lstm.reset_hidden_state()
            predictions = model(image_sequences)
        acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
        loss = cls_criterion(predictions, labels).item()
        test_metrics["loss"].append(loss)
        test_metrics["acc"].append(acc)
        sys.stdout.write(
            "\rTesting -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
            % (
                batch_i,
                len(test_dataloader),
                loss,
                np.mean(test_metrics["loss"]),
                acc,
                np.mean(test_metrics["acc"]),
            )
        )
    print("")
    model.train()
    return np.mean(test_metrics["loss"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/UCF-101-frames")
    parser.add_argument("--split_path", type=str, default="data/ucfTrainTestlist")
    parser.add_argument("--split_number", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=40)
    parser.add_argument("--img_dim", type=int, default=224)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--checkpoint_model", type=str, default="")
    parser.add_argument("--checkpoint_interval", type=int, default=5)
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    image_shape = (opt.channels, opt.img_dim, opt.img_dim)

    train_dataset = Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=True,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    test_dataset = Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=False,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    cls_criterion = nn.CrossEntropyLoss().to(device)

    model = ConvLSTM(
        num_classes=train_dataset.num_classes,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    ).to(device)

    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model))

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=7, min_delta=1e-4)

    for epoch in range(1, opt.num_epochs + 1):
        epoch_metrics = {"loss": [], "acc": []}
        prev_time = time.time()
        print(f"--- Epoch {epoch} —“)
        for batch_i, (X, y) in enumerate(train_dataloader):
            if X.size(0) == 1:
                continue
            image_sequences = Variable(X.to(device), requires_grad=True)
            labels = Variable(y.to(device), requires_grad=False)
            optimizer.zero_grad()
            model.lstm.reset_hidden_state()
            predictions = model(image_sequences)
            loss = cls_criterion(predictions, labels)
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            loss.backward()
            optimizer.step()
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["acc"].append(acc)
            batches_done = epoch * len(train_dataloader) + batch_i
            batches_left = opt.num_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)] ETA: %s"
                % (
                    epoch,
                    opt.num_epochs,
                    batch_i,
                    len(train_dataloader),
                    loss.item(),
                    np.mean(epoch_metrics["loss"]),
                    acc,
                    np.mean(epoch_metrics["acc"]),
                    time_left,
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_loss = test_model(model, test_dataloader, cls_criterion, device, epoch)
        scheduler.step(val_loss)
        early_stopping(val_loss)

        if epoch % opt.checkpoint_interval == 0:
   		os.makedirs("model_checkpoints", exist_ok=True)
    		ckpt_path = f"model_checkpoints/{model.__class__.__name__}_{epoch}.pth"
    		torch.save(model.state_dict(), ckpt_path)
    		print(f"\nSaved checkpoint: {ckpt_path}")

        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

if __name__ == "__main__":
    main()
