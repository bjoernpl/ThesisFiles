import os
import torch
from skimage import io
from torch.utils.data import Dataset
import _pickle as pickle
from pathlib import Path
import numpy as np
from PIL import Image


class FaceStyleDataset(Dataset):
    """Face Dataset"""

    def __init__(self, embeddings_file, image_dir, transform=None):
        """
        Args:
            embeddings_file (str): Path to embeddings list
            image_dir (str): Path to root of image directory
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(embeddings_file, 'rb') as f:
            self.embeddings_list = pickle.load(f)
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.embeddings_list)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        speaker, utterance, embed = self.embeddings_list[index]
        image_path = os.path.join(self.image_dir,
                                  speaker, self.utt_to_im(utterance))
        if not Path(image_path).exists():
            image_path = Path(os.path.join(self.image_dir, speaker))
            try:
                image_path = next(image_path.glob("*.jpg"))
            except StopIteration:
                print(image_path, utterance)

        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            
            image = Image.open(next(image_path.glob("*.jpg")))

        if self.transform:
            image = self.transform(image)
        
        
        return image, embed

    def utt_to_im(self, utterance):
        name = utterance.split(".")[0]
        # some embeddings for audio of speaker 15 and 18 exist twice
        try:
            name, counter = name.split("_")[:2]
        except ValueError:
            counter = 0
        return f"{name}_{int(counter)+1}.jpg"