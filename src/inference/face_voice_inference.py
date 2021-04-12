import yaml
from yaml import Loader
import os
from models.inception_resnet_v1 import InceptionResnetV1
from models.facevoice import FaceVoice
from loaders.FaceStyleLoader import FaceStyleLoader
from loaders.FaceStyleDataset import FaceStyleDataset
from parallel_wavegan.utils import load_model, download_pretrained_model
import pytorch_lightning as pl
from pathlib import Path
import torch
from inference.tacotron_loader import TacotronLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from models.mtcnn import fixed_image_standardization
from models.facenet import FaceNet
from facenet_pytorch import MTCNN


class FaceStyleInference():

    def __init__(
        self, 
        facevoice_model_path: Path,
        facenet_pretrained: str = "casia-webface",
        facenet_finetuned = None
        ):
        face_model = InceptionResnetV1(pretrained=facenet_pretrained, num_classes=5089)
        if facenet_finetuned is not None:
            face_model = FaceNet.load_from_checkpoint(facenet_finetuned, model=face_model)
        _ = face_model.eval()
        self.facestyle = FaceVoice.load_from_checkpoint(facevoice_model_path, model=face_model).cuda()
        _ = self.facestyle.eval()
        print(f"- Loaded Facestyle model from : {facevoice_model_path}")
        if facenet_finetuned is not None:
            print(f"  using finetuned Facenet model from {facenet_finetuned}")
            
        self.tacotron, self.preprocess_text = TacotronLoader.load()
        print(f"- Loaded Tacotron2 model")

        self.vocoder = load_model(
                download_pretrained_model("vctk_parallel_wavegan.v1")
            ).to("cuda").eval()
        self.vocoder.remove_weight_norm()
        print("- Loaded ParallelWaveGAN")
        
        self.mtcnn = MTCNN(margin=10)

        self.image_transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])


    def load_image(self, path):
        img = Image.open(path)
        width, height = img.size
        if (width, height) != (160, 160):
            return self.mtcnn(img).cuda()
        else:
            return self.image_transform(img).cuda()
        

    def inference(self, text, face_image, **kwargs):
        assert isinstance(text, str)

        text = self.preprocess_text(text)
        if isinstance(face_image, torch.Tensor):
            img = self.image_transform(face_image).cuda()
        elif isinstance(face_image, (os.PathLike, str)):
            img = self.load_image(face_image)
        else:
            raise TypeError("face_image must be of type torch.Tensor containing \
                image data or of str or os.Pathlike object pointing to an image file")
        with torch.no_grad():
            style = self.facestyle.forward(img.unsqueeze(0))
            c, stop_prob, alignment = self.tacotron.tts.inference(text, style=style, use_att_constraint=True, **kwargs) 
            return self.vocoder.inference(c).view(-1).cpu().numpy(), img , c, stop_prob, alignment
