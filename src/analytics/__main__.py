import os
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import transforms, datasets

from analytics.models import Input, Model
from analytics.trainers import Trainer
from analytics.metrics import Mean, PSNR, SSIM, NCC

from analytics.trainers.checkpoints import CheckpointManager
from analytics.trainers.callbacks import Callback
from analytics.trainers.training_report import TrainRepport


class Encoder(Model):

    def __init__(self, **kwarg):
        super().__init__(**kwarg)
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 7)

    def forward(self, x):
        """Forward encoding"""
        x = F.relu(self.conv1(x))  # [B, 16, 14, 14]
        x = F.relu(self.conv2(x))  # [B, 32,  7,  7]
        x = F.relu(self.conv3(x))  # [B, 64,  1,  1]
        return x


class Decoder(Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transpose1 = nn.ConvTranspose2d(64, 32, 7)
        self.transpose2 = nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1)
        self.transpose3 = nn.ConvTranspose2d(16, 1, 3, 2, 1, output_padding=1)

    def forward(self, x):
        """Forward decoding"""
        x = F.relu(self.transpose1(x))  # [B, 32,  7,  7]
        x = F.relu(self.transpose2(x))  # [B, 16, 14, 14]
        x = F.sigmoid(self.transpose3(x))  # [B, 1, 28, 28]
        return x


class AutoEncoder(Model):

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoencoderTrainer(Trainer):
    """Implementation of autoencoder training process"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._log_format:
            self._log_format = (
                "{epoch:5d} / {n_epochs:5d}"
                " \t \033[92m{message:12s}\033[0m"
                " \t {losses_Mean:14.12f}"
                " \t {metrics_PSNR:12.3f}"
                " \t {metrics_SSIM:12.3f}"
                " \t {metrics_NCC:12.3f}"
                # " \t {vl_loss:14.12f}"
                # " \t {vl_psnr:12.3f}"
            )

        self.criterion = None
        self.optimizer = None
        self.model = None

        self._train_mean_loss = Mean(name="mean_loss")
        self._valid_mean_loss = Mean(name="mean_loss")

        # self._train_psnr = PSNR(name="PSNR")
        # self._valid_psnr = PSNR(name="PSNR")
        #
        # self._train_ssim = SSIM(name="SSIM")
        # self._valid_ssim = SSIM(name="SSIM")
        #
        # self._train_ncc = NCC(name="SSIM")
        # self._valid_ncc = NCC(name="SSIM")

        self.metric_classes = {
            'losses': [Mean],
            'metrics': [PSNR, SSIM, NCC]
        }

    def compile(self, model, criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model.to(self._device)
        if self.checkpoint:
            self.checkpoint.put('optimizer', self.optimizer)
            self.checkpoint.put('model', self.model)

    def on_start_train(self):
        """We turn model in training mode"""
        self.model.train()

    def train_step(self, *args):
        """Training method on one batch"""
        batch_x = args[0]
        batch_x = batch_x.to(self._device)
        recon = self.model(batch_x)
        loss = self.criterion(recon, batch_x)
        # self._train_psnr.update_state(recon, batch_x)
        # self._train_ssim.update_state(recon, batch_x)
        # self._train_ncc.update_state(recon, batch_x)
        # self._train_mean_loss.update_state(loss.item())
        self.compute_metric('losses', loss.item())
        self.compute_metric('metrics', recon, batch_x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # return {'loss': self._train_mean_loss.result(),
        #         # 'psnr': self._train_psnr.result(),
        #         # 'ssim': self._train_ssim.result(),
        #         # 'ncc': self._train_ncc.result()
        #         }

    def estimate_step(self, *args):
        batch_x = args[0]
        batch_x = batch_x.to(self._device)
        recon = self.model(batch_x)
        loss = self.criterion(recon, batch_x)
        self._train_mean_loss.update_state(loss.item())
        loss.backward()
        return {'loss': self._train_mean_loss.result()}

    def optimize_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def on_start_eval(self):
        """We turn model in training mode"""
        self.model.eval()

    def eval_step(self, *args):
        """Training method on one batch"""
        batch_x = args[0]
        batch_x = batch_x.to(self._device)
        recon = self.model(batch_x)
        loss = self.criterion(recon, batch_x)
        # self._valid_psnr.update_state(recon, batch_x)
        # self._valid_ssim.update_state(recon, batch_x)
        # self._valid_ncc.update_state(recon, batch_x)
        # self._valid_mean_loss.update_state(loss.item())
        self.compute_metric('losses', loss.item())
        self.compute_metric('metrics', recon, batch_x)

        # return {'loss': self._valid_mean_loss.result(),
        #         # 'psnr': self._valid_psnr.result(),
        #         # 'ssim': self._valid_ssim.result(),
        #         # 'ncc': self._valid_ncc.result()
        #         }

    def run(self, *args, **kwargs):
        print(f"{'Epochs':>13s}"
              f" \t \033[92m{'Message':12s}\033[0m"
              f" \t {'Loss':>14s}"
              f" \t {'PSNR':>12s}"
              f" \t {'SSIM':>12s}"
              f" \t {'NCC':>12s}"
              # f" \t {'Vl Loss':>14s}"
              # f" \t {'Vl PSNR':>12s}"
              )
        return super().run(*args, **kwargs)


class SaveModelOnTrainEnd(Callback):

    def __init__(self, trainer, model_fp):
        super().__init__(trainer)
        self._model_fp = model_fp

    def on_train_end(self):
        """Method will be executed when the training is done"""
        weights = self.trainer.model.state_dict()
        torch.save(weights, self._model_fp)


def main():
    """Main function"""
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(root='./dataset',
                                train=True,
                                download=True,
                                transform=transform)
    # data_loader = DataLoader(dataset=mnist_data,
    #                          batch_size=64,
    #                          shuffle=True)

    # dataiter = iter(data_loader)
    # images, labels = next(dataiter)
    # print(f"images shape: {images.size()},",
    #       f" labels shapes: {labels.size()}")
    input_x = Input(shape=(1, 28, 28))
    encoder = Encoder()
    decoder = Decoder()
    model = AutoEncoder(encoder, decoder, inputs=input_x)
    model.summary(batch_size=32)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    checkpoint = CheckpointManager("autoencoder", "outputs/checkpoint_dir/")
    trainer = AutoencoderTrainer(train_dataset=mnist_data,
                                 checkpoint=checkpoint,
                                 batch_size=32)

    save_model = SaveModelOnTrainEnd(trainer, "outputs/autoencoder.pt")
    report = TrainRepport(trainer, outputs_dir="outputs/runs/")
    trainer.callbacks.append(save_model)
    trainer.callbacks.append(report)

    trainer.compile(model, criterion, optimizer)
    results = trainer.run(n_epochs=20)

    print(results)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Goodbye!")
        os.sys.exit(125)
