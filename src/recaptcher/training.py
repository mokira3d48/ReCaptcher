"""Transformer model of language training"""
import os
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchinfo import summary
# from tokenizers import Tokenizer

from analytics.trainers import Trainer
from analytics.trainers.callbacks import Callback
from analytics.trainers.checkpoints import CheckpointManager
from analytics.trainers.training_report import TrainRepport
from analytics.metrics import Mean, CER, Value

# from .models.transformer import Transformer, CosineWarmupScheduler
from recaptcher.dataset import CaptchaDataset
from recaptcher.models.tokenizer import CharacterTokenizer
from recaptcher.models.resnet_gru import CRNN, CTCLoss


LOG = logging.getLogger(__name__)


class CTCCER(CER):
    """Calcul du CER pour un algorithm CTC"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def reconstruct(cls, labels, blank=0):
        new_labels = []
        # merge same labels
        previous = None
        for l in labels:
            if l != previous:
                new_labels.append(l)
                previous = l

        # delete blank
        new_labels = [l for l in new_labels if l != blank]
        return new_labels

    @classmethod
    def ctc_greedy_decode(cls, log_prob, blank=0, **kwargs):
        intseqs = []
        labels = np.argmax(log_prob, axis=-1)
        # print(labels.size())
        for label in labels:
            intseq = cls.reconstruct(label, blank=blank)
            intseqs.append(intseq)

        return intseqs

    def compute(self, logits):
        inference = self.__class__
        log_probs = F.log_softmax(logits, dim=-1)
        intseqs = inference.ctc_greedy_decode(log_probs.detach().numpy())
        # intseqs = torch.tensor(intseqs, dtype=torch.long)
        intseqs = [torch.tensor(seq, dtype=torch.long)
                   for seq in intseqs]
        return intseqs

    def update_state(self, y_pred, y_true, lengths):
        y_pred = self.compute(y_pred)
        targets = []
        for target, length in zip(y_true, lengths):
            targets.append(target[:length])

        # print(y_pred)
        # print()
        # print(targets)
        super().update_state(y_pred, targets)


class CRNNTrainer(Trainer):
    """Implementation of autoencoder training process"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._log_format:
            self._log_format = (
                "{epoch:5d} / {n_epochs:5d}"
                " \t \033[92m{message:12s}\033[0m"
                " \t \033[93m{losses_Mean:12.4f}\033[0m"
                " \t \033[95m{metrics_CTCCER:7.4f}\033[0m"
            )

        self.metric_classes = {
            'losses': [Mean],
            'metrics': [CTCCER],
        }

        self.criterion = None
        self.optimizer = None
        self.model = None

    def compile(self, model, criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model.to(self.device)
        if self.checkpoint:
            self.checkpoint.put('optimizer', self.optimizer)
            self.checkpoint.put('model', self.model)

    def _estimate(self, batch_x, batch_y, lengths):
        batch_x = batch_x.to(self.device)  # B x 3 x H x W
        batch_y = batch_y.to(self.device)  # B x T
        lengths = lengths.to(self.device)  # B x 1

        # etape de prediction:
        outputs = self.model(batch_x)  #: B x (y_length - 1) x VOCAB_SIZE

        loss = self.criterion(outputs, batch_y, lengths)
        self.compute_metric('losses', loss.item())

        # outputs = torch.argmax(outputs, dim=-1)
        # outputs = outputs.to(torch.long)
        self.compute_metric('metrics', outputs, batch_y, lengths)
        return loss

    def on_start_train(self):
        """We turn model in training mode"""
        self.model.train()

    def train_step(self, batch_x, batch_y, lengths):
        """Training method on one batch"""
        loss = self._estimate(batch_x, batch_y, lengths)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def on_start_eval(self):
        """We turn model in eval mode"""
        self.model.eval()

    def eval_step(self, batch_x, batch_y, lengths):
        """Validation method on one batch"""
        self._estimate(batch_x, batch_y, lengths)

    def run(self, *args, **kwargs):
        print(f"{'Epochs':>13s}"
              f" \t \033[92m{'Message':12s}\033[0m"
              f" \t \033[93m{'CTCLoss':>12s}\033[0m"
              f" \t \033[95m{'CER':>7s}\033[0m"
              # f" \t {'SSIM':>12s}"
              # f" \t {'NCC':>12s}"
              # f" \t {'Vl Loss':>14s}"
              # f" \t {'Vl PSNR':>12s}"
        )
        return super().run(*args, **kwargs)


class SavedModelCallback(Callback):

    def __init__(self, model_file_path, saving_step, **kwargs):
        super().__init__(**kwargs)
        self.model_file_path = model_file_path
        self.saving_step = saving_step
        self.n_epochs = 0

        dir_name = os.path.dirname(model_file_path)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    def on_epoch_end(self):
        """Method will be executed on epoch end"""
        self.n_epochs += 1
        if self.n_epochs % self.saving_step == 0:
            weights = self.trainer.model.state_dict()
            torch.save(weights, self.model_file_path)


class Main(object):

    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.n_epochs = kwargs['n_epochs']
        self.report_dir = kwargs['report_dir']
        self.saved_model = kwargs['savedModel']
        self.saving_step = kwargs['savingStep']

        # self.label_smoothing_eps = kwargs['LabelSmoothing']['epsilon']

        self.learning_rate = kwargs['optim']['learning_rate']
        self.betas = kwargs['optim']['beta']
        self.eps = kwargs['optim']['eps']
        # self.warmup = kwargs['optim']['warmup']
        # self.lrs_max_iters = kwargs['optim']['max_iters']

        self.train_ds = kwargs['dataset']['train']
        self.valid_ds = kwargs['dataset']['valid']
        self.test_ds = kwargs['dataset']['test']
        # self.src_column_name = kwargs['dataset']['src_column_name']
        # self.tgt_column_name = kwargs['dataset']['tgt_column_name']
        self.batch_size = kwargs['dataset']['batch_size']
        self.n_workers = kwargs['dataset']['n_workers']

        self.model_config_fp = kwargs['model']['configFile']
        self.vocab_file = kwargs['model']['vocab_file']

        self.folder = kwargs['checkpoint']['folder']
        self.last_ckp_count = kwargs['checkpoint']['last_ckp_count']

    def run(self):
        """Method to run training loop"""
        tokenizer = CharacterTokenizer()
        tokenizer.load_from_file(self.vocab_file)

        # model = Transformer.from_config_file(
        #     config_fp=self.model_config_fp,
        #     src_vocab_size=src_vocab_size,
        #     tgt_vocab_size=tgt_vocab_size,
        # )
        model = CRNN(in_channels=1, n_chars=tokenizer.vocab_size)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        x = torch.randn(self.batch_size, 1, 224, 224)
        summary(model, input_data=x)

        train_dataset = CaptchaDataset(self.train_ds)
        valid_dataset = CaptchaDataset(self.valid_ds)

        criterion = CTCLoss(blank_index=tokenizer.pad)
        # criterion = LabelSmoothingLoss(epsilon=self.label_smoothing_eps)
        optimizer = optim.Adam(model.parameters(),
                               lr=self.learning_rate,
                               betas=self.betas,
                               eps=self.eps)
        # lr_scheduler = CosineWarmupScheduler(optimizer=optimizer,
        #                                      warmup=self.warmup,
        #                                      max_iters=self.lrs_max_iters)

        checkpoint = CheckpointManager(self.name,
                                       self.folder,
                                       last_ckp_max_count=self.last_ckp_count)
        trainer = CRNNTrainer(train_dataset=train_dataset,
                              valid_dataset=valid_dataset,
                              checkpoint=checkpoint,
                              batch_size=self.batch_size,
                              n_workers=self.n_workers)

        report = TrainRepport(trainer, outputs_dir=self.report_dir)
        saving = SavedModelCallback(self.saved_model,
                                    self.saving_step,
                                    trainer=trainer)
        trainer.callbacks.append(report)
        trainer.callbacks.append(saving)

        trainer.compile(model, criterion, optimizer)
        trainer.run(self.n_epochs)
