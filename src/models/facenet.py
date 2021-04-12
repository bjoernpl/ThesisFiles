import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from loss.focal_loss import FocalLoss

class FaceNet(pl.LightningModule):

    def __init__(self, model, learning_rate=None, lr_decay=0.1):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.fl = FocalLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # x = faces (batch_size, 3, 160, 160)
        # y = style_emb (batch_size, 512)
        x, y = batch
        y_pred = self(x)
        loss = self.fl(y_pred, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.fl(y_pred, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.fl(y_pred, y)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    
    def configure_optimizers(self):
        optimizer = Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        scheduler = MultiStepLR(
            optimizer, 
            milestones=[3], 
            gamma=self.lr_decay
        )
        return [optimizer], [scheduler]
