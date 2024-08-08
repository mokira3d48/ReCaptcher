"""Transformer model of language training"""
import os
import logging
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
# from .dataset import BilingualDataset

LOG = logging.getLogger(__name__)


class LabelSmoothingLoss(nn.Module):

    def __init__(self, epsilon=0.1, reduction="mean", weight=None):
        super().__init__()
        assert 0 <= epsilon < 1

        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, i, j):
        return (1 - self.epsilon) * i + self.epsilon * j

    def forward(self, predict, target):
        """Compute loss label smoothed"""
        if self.weight is not None:
            self.weight = self.weight.to(predict)

        num_classes = predict.size(-1)
        # log_probs = torch.log_softmax(predict, dim=-1)
        log_probs = predict
        loss = self.reduce_loss(-log_probs.sum(dim=-1))

        negative_log_likelihood_loss = F.nll_loss(
            input=log_probs,
            target=target,
            reduction=self.reduction,
            weight=self.weight,
        )
        returned = self.linear_combination(negative_log_likelihood_loss,
                                           (loss / num_classes))
        return returned


class TransformerTrainer(Trainer):
    """Implementation of autoencoder training process"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._log_format:
            self._log_format = (
                "{epoch:5d} / {n_epochs:5d}"
                " \t \033[92m{message:12s}\033[0m"
                " \t \033[93m{losses_Mean:12.6f}\033[0m"
                " \t \033[95m{metrics_CER:8.6f}\033[0m"
                " \t \033[96m{lr_Value:8.6f}\033[0m"
            )

        self.metric_classes = {
            'losses': [Mean],
            'metrics': [CER],
            'lr': [Value],
        }

        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.model = None

    def compile(self, model, criterion, optimizer, scheduler):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model.to(self.device)
        if self.checkpoint:
            self.checkpoint.put('optimizer', self.optimizer)
            self.checkpoint.put('scheduler', self.scheduler)
            self.checkpoint.put('model', self.model)

    def _estimate(self, batch_x, batch_y):
        batch_x = batch_x.to(self.device)  # B x 3 x H x W
        batch_y = batch_y.to(self.device)  # B x T

        input_y = batch_y[:, :-1]  # 1 x (y_length - 1)
        target = batch_y[:, 1:]  # 1 x (y_length - 1)
        # latent = self.model.encode(batch_x)
        # probs = self.model.decode(latent, input_y)
        probs = self.model(batch_x, input_y)
        # print(torch.argmax(torch.log_softmax(probs, dim=-1), dim=-1))

        predicts = probs.contiguous().view(-1, self.model.tgt_vocab_size)
        targets = target.contiguous().view(-1)
        loss = self.criterion(predicts, targets)
        self.compute_metric('losses', loss.item())

        output = torch.argmax(probs, dim=-1)
        output = output.to(torch.long)
        self.compute_metric('metrics', output, target)

        return loss

    def on_start_train(self):
        """We turn model in training mode"""
        self.model.train()

    def train_step(self, batch_x, batch_y):
        """Training method on one batch"""
        loss = self._estimate(batch_x, batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        lr = self.scheduler.get_lr()
        self.compute_metric('lr', lr[0])

    def on_start_eval(self):
        """We turn model in training mode"""
        self.model.eval()

    def eval_step(self, batch_x, batch_y):
        """Validation method on one batch"""
        self._estimate(batch_x, batch_y)

    def run(self, *args, **kwargs):
        print(f"{'Epochs':>13s}"
              f" \t \033[92m{'Message':12s}\033[0m"
              f" \t \033[93m{'Loss':>12s}\033[0m"
              f" \t \033[95m{'CER':>8s}\033[0m"
              f" \t \033[96m{'LR':>8s}\033[0m"
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

        self.label_smoothing_eps = kwargs['LabelSmoothing']['epsilon']

        self.learning_rate = kwargs['optim']['learning_rate']
        self.betas = kwargs['optim']['beta']
        self.eps = kwargs['optim']['eps']
        self.warmup = kwargs['optim']['warmup']
        self.lrs_max_iters = kwargs['optim']['max_iters']

        self.train_ds = kwargs['dataset']['train']
        self.valid_ds = kwargs['dataset']['valid']
        self.test_ds = kwargs['dataset']['test']
        self.src_column_name = kwargs['dataset']['src_column_name']
        self.tgt_column_name = kwargs['dataset']['tgt_column_name']
        self.batch_size = kwargs['dataset']['batch_size']
        self.n_workers = kwargs['dataset']['n_workers']

        self.model_config_fp = kwargs['model']['configFile']
        self.src_vocab = kwargs['model']['src_vocab']
        self.tgt_vocab = kwargs['model']['tgt_vocab']

        self.folder = kwargs['checkpoint']['folder']
        self.last_ckp_count = kwargs['checkpoint']['last_ckp_count']

    def run(self):
        """Method to run training loop"""
        src_tokenizer = Tokenizer.from_file(self.src_vocab)
        tgt_tokenizer = Tokenizer.from_file(self.tgt_vocab)
        src_vocab_size = len(src_tokenizer.get_vocab())
        tgt_vocab_size = len(tgt_tokenizer.get_vocab())

        model = Transformer.from_config_file(
            config_fp=self.model_config_fp,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        x = torch.randint(
            low=0,
            high=(src_vocab_size - 1),
            size=(self.batch_size, model.src_seq_max_len),
        )
        y_shift = torch.randint(
            low=0,
            high=(tgt_vocab_size - 1),
            size=(self.batch_size, model.tgt_seq_max_len)
        )
        summary(model, input_data=[x, y_shift])

        train_dataset = BilingualDataset(
            self.train_ds,
            self.src_column_name,
            self.tgt_column_name,
            src_tokenizer,
            tgt_tokenizer,
            src_seq_max_len=model.src_seq_max_len,
            tgt_seq_max_len=model.tgt_seq_max_len
        )
        valid_dataset = BilingualDataset(
            self.valid_ds,
            self.src_column_name,
            self.tgt_column_name,
            src_tokenizer,
            tgt_tokenizer,
            src_seq_max_len=model.src_seq_max_len,
            tgt_seq_max_len=model.tgt_seq_max_len
        )

        print("Source lang:", train_dataset.src_column_name)
        print("Target lang:", train_dataset.tgt_column_name)
        print()

        criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.tgt_pad)
        # criterion = LabelSmoothingLoss(epsilon=self.label_smoothing_eps)
        optimizer = optim.Adam(model.parameters(),
                               lr=self.learning_rate,
                               betas=self.betas,
                               eps=self.eps)
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer,
                                             warmup=self.warmup,
                                             max_iters=self.lrs_max_iters)

        checkpoint = CheckpointManager(self.name,
                                       self.folder,
                                       last_ckp_max_count=self.last_ckp_count)
        trainer = TransformerTrainer(train_dataset=train_dataset,
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

        trainer.compile(model, criterion, optimizer, lr_scheduler)
        # trainer.compile(model, criterion, optimizer)
        trainer.run(self.n_epochs)
