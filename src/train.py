import os
from pathlib import Path

import pytorch_lightning as pl
import yaml
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from yaml import Loader

from loaders.FaceStyleDataset import FaceStyleDataset
from loaders.FaceStyleLoader import FaceStyleLoader
from models.facenet import FaceNet
from models.facevoice import FaceVoice


def main(params):
    # Load pretrained facenet module
    facenet_model = InceptionResnetV1(pretrained=params["pretrained_facenet"])
    if "finetuned_facenet_path" in params:
        facenet_model = InceptionResnetV1(pretrained=params["pretrained_facenet"], num_classes=5089)
        facenet_model = FaceNet.load_from_checkpoint(params["finetuned_facenet_path"], model=facenet_model)

    # set freeze parameters
    include = params["include_layers"]
    for name, param in facenet_model.named_parameters():
        print(name)
        if name not in include:
            param.requires_grad = False

    # load FaceVoice model
    facevoice = FaceVoice(
        model = facenet_model,
        learning_rate = params["learning_rate"],
        lr_decay = params["lr_decay"]
    )

    # Load DataModule
    data = FaceStyleLoader(
        data_root = Path(params["data_root"]),
        batch_size = params["batch_size"]
    )

    # Weights and biases logging
    wandb_logger = pl.loggers.WandbLogger(
        project = params["wandb_project"]
    )
    wandb_logger.watch(facevoice, log='all')
    wandb_logger.log_hyperparams(params)
    wandb_logger.experiment.save("src/*")
    
    # Saves 4 best checkpoints
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'results/{wandb_logger.experiment.name}/',
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
    trainer.fit(facevoice, datamodule=data)
    trainer.test(facevoice, datamodule=data)


if __name__ == "__main__":
    config_path = "src/conf.yaml"
    assert os.path.exists(config_path)

    with open(config_path, 'r') as f:
        params = yaml.load(f, Loader=Loader)
    main(params)
