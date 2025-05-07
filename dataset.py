import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class Dataset(Dataset):
    def __init__(self, dataset_path, split_path, split_number, input_shape, sequence_length, training):
        self.training = training
        self.label_index = self._extract_label_mapping(split_path)
        self.sequences = self._extract_sequence_paths(dataset_path, split_path, split_number, training)
        self.sequence_length = sequence_length
        self.label_names = sorted(list(set([self._activity_from_path(seq_path) for seq_path in self.sequences])))
        self.num_classes = len(self.label_names)
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _extract_label_mapping(self, split_path="data/ucfTrainTestlist"):
        with open(os.path.join(split_path, "classInd.txt")) as file:
            lines = file.read().splitlines()
        label_mapping = {}
        for line in lines:
            label, action = line.split()
            label_mapping[action] = int(label) - 1
        return label_mapping

    def _extract_sequence_paths(self, dataset_path, split_path="data/ucfTrainTestlist", split_number=1, training=True):
        assert split_number in [1, 2, 3], "Split number must be 1, 2, or 3"
        fn = f"trainlist0{split_number}.txt" if training else f"testlist0{split_number}.txt"
        with open(os.path.join(split_path, fn)) as file:
            lines = file.read().splitlines()
        return [os.path.join(dataset_path, line.split(".avi")[0]) for line in lines]

    def _activity_from_path(self, path):
        return path.split("/")[-2]

    def _frame_number(self, image_path):
        return int(image_path.split("/")[-1].split(".jpg")[0])

    def _pad_to_length(self, sequence):
        if self.sequence_length is not None and len(sequence) < self.sequence_length:
            pad_count = self.sequence_length - len(sequence)
            sequence = [sequence[0]] * pad_count + sequence
        return sequence

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        image_paths = sorted(glob.glob(f"{sequence_path}/*.jpg"), key=self._frame_number)
        image_paths = self._pad_to_length(image_paths)

        if self.training:
            max_interval = max(1, len(image_paths) // self.sequence_length)
            sample_interval = np.random.randint(1, max_interval + 1)
            max_start = max(1, len(image_paths) - sample_interval * self.sequence_length + 1)
            start_i = np.random.randint(0, max_start)
            flip = np.random.random() < 0.5
        else:
            start_i = 0
            sample_interval = 1 if self.sequence_length is None else max(1, len(image_paths) // self.sequence_length)
            flip = False

        image_sequence = []
        for i in range(start_i, len(image_paths), sample_interval):
            if self.sequence_length is None or len(image_sequence) < self.sequence_length:
                img = Image.open(image_paths[i]).convert("RGB")
                image_tensor = self.transform(img)
                if flip:
                    image_tensor = torch.flip(image_tensor, dims=(-1,))
                image_sequence.append(image_tensor)

        image_sequence = torch.stack(image_sequence).float()
        target = self.label_index[self._activity_from_path(sequence_path)]
        return image_sequence, target

    def __len__(self):
        return len(self.sequences)
