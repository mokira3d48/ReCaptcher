import logging

import numpy as np
import cv2 as cv
import torch
from torch import nn
from torch.nn import functional as F
from analytics.models import Model


IMAGE_WIDTH = 112
IMAGE_HEIGHT = 112
N_CHANNELS = 3

LOG = logging.getLogger(__name__)


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        identify_downsample=None,
        stride = 1,
    ):
        super().__init__()
        self.identify_downsample = identify_downsample
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2= nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            (out_channels*self.expansion),
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward computation Method"""
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identify_downsample is not None:
            identity = self.identify_downsample(identity)

        x = x + identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, num_layers, image_channels):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=image_channels,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers:
        self.layer1 = self._make_layer(block=block,
                                       num_residual_blocks=num_layers[0],
                                       out_channels=64,
                                       stride=1)
        self.layer2 = self._make_layer(block=block,
                                       num_residual_blocks=num_layers[1],
                                       out_channels=128,
                                       stride=2)
        self.layer3 = self._make_layer(block=block,
                                       num_residual_blocks=num_layers[2],
                                       out_channels=256,
                                       stride=2)
        self.layer4 = self._make_layer(block=block,
                                       num_residual_blocks=num_layers[3],
                                       out_channels=512,
                                       stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512*4, num_classes)

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        """Method to build a ResNet Layer"""
        identify_downsample = None
        layers = []

        if stride != 1 or self.in_channels != (4*out_channels):
            identify_downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                          out_channels=(4*out_channels),
                          kernel_size=1,
                          stride=stride),
                nn.BatchNorm2d(4*out_channels),
            )

        layers.append(
            block(self.in_channels, out_channels, identify_downsample, stride)
        )
        self.in_channels = 4*out_channels
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        return x


class Encoder(Model):

    def __init__(self, in_channels, resnet_version=50):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = None
        if resnet_version == 50:
            self.conv_layers = self.create_resnet([3, 4, 6, 3])

    def create_resnet(self, layers):
        return ResNet(ResidualBlock, layers, self.in_channels)

    @classmethod
    def load_config(cls, in_channels=3, resnet_version=50):
        return cls(in_channels=in_channels, resnet_version=resnet_version)

    def forward(self, x):
        x = self.conv_layers(x)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 num_chars,
                 hidden_size=512,
                 num_layers=2,
                 dropout_prob=0.2):
        super().__init__()
        self.gru = nn.GRU(2048,
                          hidden_size,
                          bidirectional=True,
                          num_layers=num_layers,
                          dropout=dropout_prob)
        self.fully_connected = nn.Linear(2*hidden_size, num_chars)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # NOTE: +1 to n_chars for the blank char.
        # 2*hidden_size if bidirectional==True else hidden_size.

        # weight initialization:
        self.fully_connected.weight.data.normal_(0., 0.02)
        self.fully_connected.bias.data.fill_(0)

    def forward(self, x):
        n_batch, n_channels, height, width = x.size()

        x = x.view(n_batch, n_channels, -1).contiguous()  # [B, 2048, 49]
        x = x.permute(0, 2, 1).contiguous()  # [B, 49, 2048]

        x, _ = self.gru(x)
        x = self.fully_connected(x)  # [None, 49, 65]
        x = self.log_softmax(x)  # [None, 49, 65]
        return x


class CRNN(nn.Module):  # noqa

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoding = encoder
        self.decoding = decoder

    def forward(self, x):
        feature_maps = self.encoding(x)
        outputs = self.decoding(feature_maps)
        return outputs


class CTCLoss(nn.Module):

    def __init__(self, blank_index=0):
        super().__init__()
        LOG.info("CTC BLANK INDEX: " + str(blank_index))
        self.ctc_loss = nn.CTCLoss(blank=blank_index)

    def forward(self, log_probs, targets, lengths):
        B, T, C = log_probs.size()  # noqa
        log_probs = log_probs.permute(1, 0, 2)  # T B C
        logit_lengths = torch.LongTensor([log_probs.size(0)]*B)
        target_lengths = lengths
        loss = self.ctc_loss(log_probs,
                             targets,
                             logit_lengths,
                             target_lengths,)
        return loss


class InputFunction(nn.Module):

    def __init__(self, image_preprocess):
        super().__init__()
        self.image_preprocess = image_preprocess

    def forward(self, images):
        preprocessed_images = []
        for image in images:
            image = self.image_preprocess(image)
            preprocessed_images.append(image)

        preprocessed_images = np.asarray(preprocessed_images)
        preprocessed_images = torch.tensor(preprocessed_images)
        return preprocessed_images


class CTCDecoder(nn.Module):
    """Implementation of CTC decoding"""

    def __init__(self, blank_index=0):
        super().__init__()
        self.blank_index = blank_index

    def decode_function(self, log_probs):
        """Function of CTC sequence decoding"""
        int_sequence = torch.argmax(log_probs, dim=-1)
        int_sequence = int_sequence.detach().cpu().numpy()
        # LOG.debug("integers sequence: " + str(int_sequence))
        decoded_seq = []
        preview_index = -1
        for index in int_sequence:
            if index != self.blank_index and index != preview_index:
                decoded_seq.append(index)
                preview_index = index

        return decoded_seq

    def forward(self, batch):
        sequences_decoded = []
        for log_probs in batch:
            seq_returned = self.decode_function(log_probs)
            seq_returned = torch.tensor(seq_returned, dtype=torch.long)
            sequences_decoded.append(seq_returned)

        return sequences_decoded


class OutputFunction(nn.Module):

    def __init__(self, token_decoding, blank_index=0):
        super().__init__()
        self.token_decoding = token_decoding
        self.ctc_decoding = CTCDecoder(blank_index=blank_index)

    def forward(self, log_probs):
        strings_decoded = []
        token_sequences = self.ctc_decoding(log_probs)
        for token_sequence in token_sequences:
            str_decoded = self.token_decoding(token_sequence)
            strings_decoded.append(str_decoded)

        return strings_decoded


def test():
    """Function of test"""
    from torchinfo import summary

    loss_fn = CTCLoss()
    ctc_decoder = CTCDecoder()
    encoder = Encoder(in_channels=3)
    decoder = Decoder(num_chars=64)
    model = CRNN(encoder, decoder)
    summary(model, input_size=(16, N_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))

    x = torch.randn(4, N_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    y = model(x)
    print("y=", y.size())
    print("total=", sum(y[0, 0, :]))
    print("decoded: " + str(ctc_decoder(y)))

    target = torch.randint(1, 20, (4, 5))
    loss = loss_fn(y, target, torch.tensor([5]*4))
    print("loss=", loss.item())


if __name__ == '__main__':
    test()

