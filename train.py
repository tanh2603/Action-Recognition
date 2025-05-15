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

from models import ConvLSTM
from dataset import Dataset

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

def test_model(model, test_dataloader, criterion, device):
    print("\nTesting model...")
    model.eval()
    test_metrics = {"loss": [], "acc": []}
    with torch.no_grad():
        for batch_i, (X, y) in enumerate(test_dataloader):
            image_sequences = X.to(device)
            labels = y.to(device)
            model.reset_hidden_state()
            predictions = model(image_sequences)
            loss = criterion(predictions, labels).item()
            acc = 100 * (predictions.argmax(1) == labels).cpu().numpy().mean()
            test_metrics["loss"].append(loss)
            test_metrics["acc"].append(acc)
            sys.stdout.write(
                "\rTesting -- [Batch %d/%d] [Loss: %f (%.4f), Acc: %.2f%% (%.2f%%)]"
                % (
                    batch_i + 1,
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

def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_shape = (opt.channels, opt.img_dim, opt.img_dim)

    # Dataset và dataloader
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

    # Khởi tạo model
    model = ConvLSTM(
        num_classes=train_dataset.num_classes,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    early_stopping = EarlyStopping(patience=7, min_delta=1e-4)

    start_epoch = 0
    if opt.checkpoint_model:
        print(f"Loading checkpoint from {opt.checkpoint_model} ...")
        checkpoint = torch.load(opt.checkpoint_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    # Vòng train
    for epoch in range(start_epoch, opt.num_epochs):
        epoch_metrics = {"loss": [], "acc": []}
        prev_time = time.time()
        print(f"--- Epoch {epoch + 1}/{opt.num_epochs} ---")
        for batch_i, (X, y) in enumerate(train_dataloader):
            # Bỏ qua batch size = 1 vì lỗi batchnorm/LSTM
            if X.size(0) == 1:
                continue
            image_sequences = X.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            model.reset_hidden_state()
            predictions = model(image_sequences)
            loss = cls_criterion(predictions, labels)
            acc = 100 * (predictions.argmax(1) == labels).cpu().numpy().mean()
            loss.backward()
            optimizer.step()

            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["acc"].append(acc)

            batches_done = epoch * len(train_dataloader) + batch_i
            batches_left = opt.num_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %.6f (%.6f), Acc: %.2f%% (%.2f%%)] ETA: %s"
                % (
                    epoch + 1,
                    opt.num_epochs,
                    batch_i + 1,
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

        val_loss = test_model(model, test_dataloader, cls_criterion, device)
        scheduler.step(val_loss)
        early_stopping(val_loss)

        # Lưu checkpoint định kỳ
        if (epoch + 1) % opt.checkpoint_interval == 0 or early_stopping.early_stop:
            os.makedirs("model_checkpoints", exist_ok=True)
            checkpoint_path = f"model_checkpoints/{model.__class__.__name__}_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")

        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/UCF-101-frames", help="Path to UCF-101 dataset")
    parser.add_argument("--split_path", type=str, default="data/ucfTrainTestlist", help="Path to train/test split")
    parser.add_argument("--split_number", type=int, default=1, help="Train/test split number {1, 2, 3}")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=40, help="Number of frames per sequence")
    parser.add_argument("--img_dim", type=int, default=224, help="Image height/width")
    parser.add_argument("--channels", type=int, default=3, help="Image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Latent dimension size")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Path to checkpoint to resume training")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Epoch interval to save checkpoints")
    opt = parser.parse_args()

    main(opt)
