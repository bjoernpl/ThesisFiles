import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from yaml import Loader

from loaders.FaceStyleDataset import FaceStyleDataset
from loaders.FaceStyleLoader import FaceStyleLoader
from models.facenet import FaceNet
from models.facevoice import FaceVoice
from models.inception_resnet_v1 import InceptionResnetV1
from models.mtcnn import fixed_image_standardization


def main(params):
    
    trans = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])

    # Load training dataset
    train_dataset = datasets.ImageFolder(Path(params["data_root"]) / "train", transform=trans)

    #Split dataset into train / val / test sets (70/20/10%)
    img_inds = np.arange(len(train_dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(0.7 * len(img_inds))]
    val_inds = img_inds[int(0.7 * len(img_inds)):int(0.9 * len(img_inds))]
    test_inds = img_inds[int(0.9 * len(img_inds)):]

    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        batch_size=params["batch_size"],
        sampler=SubsetRandomSampler(train_inds)
    )
    val_loader = DataLoader(
        train_dataset,
        num_workers=8,
        batch_size=params["batch_size"],
        sampler=SubsetRandomSampler(val_inds)
    )
    test_loader = DataLoader(
        train_dataset,
        num_workers=8,
        batch_size=params["batch_size"],
        sampler=SubsetRandomSampler(test_inds)
    )

    # Load pretrained facenet module
    n_classes = len(train_dataset.class_to_idx)
    facenet_model = InceptionResnetV1(pretrained=params["pretrained_facenet"], classify=True, num_classes=n_classes)
    facenet_pl = FaceNet(facenet_model, params["learning_rate"], params["lr_decay"])
    
    # set freeze parameters
    if "include_layers" in params:
        include = params["include_layers"]
        for name, param in facenet_model.named_parameters():
            if name not in include:
                param.requires_grad = False

    # Weights and biases logging
    wandb_logger = pl.loggers.WandbLogger(
        project = params["wandb_project"]
    )
    wandb_logger.watch(facenet_pl, log='all')
    wandb_logger.log_hyperparams(params)
    wandb_logger.experiment.save("src/*")
    
    # Saves 4 best checkpoints
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'results/facenet/{wandb_logger.experiment.name}/',
        filename='facestyle-{epoch:02d}-{val_loss:.2f}',
        save_top_k=4,
        mode='min',
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    
    # Trainer with data parallel accelerator
    trainer = pl.Trainer(
        gpus=2,
        accelerator='dp',
        logger=wandb_logger,
        log_every_n_steps=20,
        max_epochs=params["max_epochs"],
        callbacks=[checkpoint_callback, lr_monitor]
    )
    #trainer.tune(facevoice, datamodule=data)
    trainer.fit(facenet_pl, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.test(facenet_pl, test_dataloaders=test_loader)


if __name__ == "__main__":
    config_path = "src/conf_facenet.yaml"
    assert os.path.exists(config_path)

    with open(config_path, 'r') as f:
        params = yaml.load(f, Loader=Loader)
    main(params)
