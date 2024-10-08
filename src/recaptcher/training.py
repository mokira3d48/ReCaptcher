"""Transformer model of language training"""
import json
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
from recaptcher.models.preprocessing import CharacterTokenizer
from recaptcher.models.resnet_gru import (Encoder,
                                          Decoder,
                                          CRNN,
                                          CTCLoss,
                                          CTCDecoder)


LOG = logging.getLogger(__name__)


def load_from_file(model, file_path):
    """Function to load model from the model file path"""
    loaded = torch.load(file_path)
    model.load_state_dict(loaded)
    print("model is loaded from saving file path.")


def load_from_checkpoint(model, file_path):
    """Function to load model from checkpoint file"""
    loaded = torch.load(file_path)
    if 'model' in loaded:
        model.load_state_dict(loaded['model'])
        LOG.info("model is loaded from checkpoint file.")


class CTCCER(CER):  # noqa
    """Calcul du CER pour un algorithm CTC"""

    BLANK_INDEX = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # LOG.info("CTC CER blank index: " + str(self.__class__.BLANK_INDEX))
        self.decoder = CTCDecoder(blank_index=self.__class__.BLANK_INDEX)

    def update_state(self, y_pred, y_true, lengths):  # noqa
        y_pred = self.decoder(y_pred)  # noqa
        targets = []
        for target, length in zip(y_true, lengths):
            targets.append(target[:length])

        # LOG.debug("")
        # LOG.debug("targets: " + str(targets[0]))
        # LOG.debug("outputs: " + str(y_pred[0]))
        # input()
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


class BestModelSavingCallback(Callback):

    def __init__(self, name, dir_path, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.dir_path = os.path.join(dir_path, f'{name}_model/')
        self.best_scores = {}

        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        self.best_value_fp = os.path.join(self.dir_path, 'best.json')
        if os.path.isfile(self.best_value_fp):
            with open(self.best_value_fp, mode='r', encoding='UTF-8') as file:
                self.best_scores = json.load(file)
        else:
            self.update_scores_dict()

    def update_scores_dict(self, **kwargs):
        self.best_scores.update(kwargs)
        with open(self.best_value_fp, mode='w', encoding='UTF-8') as file:
            scores_jsonify = json.dumps(self.best_scores, indent=4)
            file.write(scores_jsonify)

    def save_model(self):
        model = self.trainer.model
        model_weights = model.state_dict()
        torch.save(model_weights, os.path.join(self.dir_path, 'best.pt'))

    def on_epoch_end(self):
        """Method will be executed on epoch end"""
        current_scores = {
            'CER': self.trainer.valid_results['metrics_CTCCER'][-1].item(),
            'CTC': self.trainer.valid_results['losses_Mean'][-1].item(),
        }
        if not self.best_scores:
            self.save_model()
            self.update_scores_dict(**current_scores)
            return

        has_best_cer = current_scores['CER'] < self.best_scores['CER']
        has_best_ctc = current_scores['CTC'] < self.best_scores['CTC']
        if has_best_cer or has_best_ctc:
            # LOG.debug("old best: " + str(self.best_scores))
            # LOG.debug("current: " + str(current_cer))
            self.save_model()
            self.update_scores_dict(**current_scores)


class Main(object):

    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.n_epochs = kwargs['n_epochs']
        self.report_dir = kwargs['report_dir']
        self.saved_model = kwargs['savedModel']
        self.model_file_path = kwargs['model_file_path']

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

        self.encoder_config_fp = kwargs['model']['decoder']['configFile']
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
        encoder = Encoder.from_config_file( self.encoder_config_fp)
        decoder = Decoder(num_chars=tokenizer.vocab_size)
        model = CRNN(encoder, decoder)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        if self.model_file_path:
            load_from_file(model, self.model_file_path)

        x = torch.randn(self.batch_size, 1, 112, 112)
        summary(model, input_data=x)

        train_dataset = CaptchaDataset(self.train_ds)
        valid_dataset = CaptchaDataset(self.valid_ds)

        CTCCER.BLANK_INDEX = tokenizer.pad_index
        criterion = CTCLoss(blank_index=tokenizer.pad_index)
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
        saving = BestModelSavingCallback(self.name,
                                         self.report_dir,
                                         trainer=trainer)
        trainer.callbacks.append(report)
        trainer.callbacks.append(saving)

        trainer.compile(model, criterion, optimizer)
        trainer.run(self.n_epochs)
        torch.save(encoder.state_dict(), 'resnet50_encoder.pt')

