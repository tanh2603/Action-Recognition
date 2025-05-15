import torch
import torch.nn as nn
import sys
import numpy as np
from models import ConvLSTM
from dataset import Dataset
from torch.utils.data import DataLoader
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/UCF-101-frames", help="Path to UCF-101 dataset")
    parser.add_argument("--split_path", type=str, default="data/ucfTrainTestlist", help="Path to train/test split")
    parser.add_argument("--split_number", type=int, default=1, help="Train/test split number. One of {1, 2, 3}")
    parser.add_argument("--img_dim", type=int, default=112, help="Height / width dimension")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    parser.add_argument("--sequence_length", type=int, default=16, help="Length of input frame sequence")
    parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_shape = (opt.channels, opt.img_dim, opt.img_dim)

    # Load test dataset
    test_dataset = Dataset(
        dataset_path=opt.dataset_path,
        split_path=opt.split_path,
        split_number=opt.split_number,
        input_shape=image_shape,
        sequence_length=opt.sequence_length,
        training=False,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Define model
    model = ConvLSTM(
        num_classes=test_dataset.num_classes,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
    )
    model = model.to(device)
    model.load_state_dict(torch.load(opt.checkpoint_model))
    model.eval()

    # Loss function
    cls_criterion = nn.CrossEntropyLoss()

    test_metrics = {"loss": []}
    test_accuracies = []

    print("Starting testing...")
    for batch_i, (X, y) in enumerate(test_dataloader):
        image_sequences = X.to(device)
        labels = y.to(device)
        with torch.no_grad():
            # Reset LSTM hidden state between sequences
            model.lstm.reset_hidden_state()
            # Forward pass
            predictions = model(image_sequences)

            loss = 0
            pred_hists = np.zeros((predictions.size(0), predictions.size(-1)))
            for t in range(opt.sequence_length):
                loss += cls_criterion(predictions[:, t], labels).item() / opt.sequence_length
                pred_hists[:, predictions[:, t].argmax(1).cpu().numpy()] += 1

        test_metrics["loss"].append(loss)
        acc = 100 * np.mean(pred_hists.argmax(1) == labels.cpu().numpy())
        test_accuracies.append(acc)

        sys.stdout.write(
            "\rTesting -- [Batch %d/%d] [Loss: %.4f (%.4f), Acc: %.2f%% (%.2f%%)]"
            % (batch_i + 1, len(test_dataloader), loss, np.mean(test_metrics["loss"]), acc, np.mean(test_accuracies))
        )
        sys.stdout.flush()

    print("\nTesting completed.")
    print(f"Average loss: {np.mean(test_metrics['loss']):.4f}")
    print(f"Average accuracy: {np.mean(test_accuracies):.2f}%")
