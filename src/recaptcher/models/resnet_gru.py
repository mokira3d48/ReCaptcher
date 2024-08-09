import logging
import torch
from torch import nn
from torch.nn import functional as F


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
                               kernel_size=7, #7
                               stride=2,  #2
                               padding=3  #3
                               )
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


class CRNN(nn.Module):

    def __init__(self, in_channels, n_chars, hidden_size=512):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_resnet([3, 4, 6, 3])
        self.gru = nn.GRU(2048,
                          hidden_size,
                          bidirectional=True,
                          num_layers=2,
                          dropout=0.2)
        self.fully_connected = nn.Linear(2*hidden_size, n_chars)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # NOTE: +1 to n_chars for the blank char.
        # 2*hidden_size if bidirectional==True else hidden_size.

        # weight initialization:
        self.fully_connected.weight.data.normal_(0., 0.02)
        self.fully_connected.bias.data.fill_(0)

    def create_resnet(self, layers):
        return ResNet(ResidualBlock, layers, self.in_channels)

    def forward(self, x):
        """Method of forward propagation"""
        x = self.conv_layers(x)  # [None, 2048, 7, 7]
        n_batch, n_channels, height, width = x.size()

        x = x.view(n_batch, n_channels, -1).contiguous()  # [B, 2048, 49]
        x = x.permute(0, 2, 1).contiguous()  # [B, 49, 2048]

        x, _ = self.gru(x)
        x = self.fully_connected(x)  # [None, 49, 65]
        x = self.log_softmax(x)  # [None, 49, 65]
        return x


class CTCLoss(nn.Module):

    def __init__(self, blank_index=0):
        super().__init__()
        LOG.info("CTC BLANK INDEX: " + str(blank_index))
        self.ctc_loss = nn.CTCLoss(blank=blank_index)

    def forward(self, log_probs, targets, lengths):
        B, T, C = log_probs.size()
        log_probs = log_probs.permute(1, 0, 2)  # T B C
        logits_lengths = torch.LongTensor([log_probs.size(0)]*B)
        target_lengths = lengths
        loss = self.ctc_loss(log_probs,
                             targets,
                             logits_lengths,
                             target_lengths,)
        return loss


class CTCDecoder(nn.Module):

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def decode_predictions(self, preds):
        # preds = preds.permute(1, 0, 2)
        preds = torch.softmax(preds, 2)
        preds = torch.argmax(preds, 2)
        preds = preds.detach().cpu().numpy()
        print(preds.shape)
        cap_preds = []
        for i in range(preds.shape[0]):
            temp = []
            for k in preds[i, :]:
                k = k - 1
                if k == 0:
                    temp.append("_")
                else:
                    decoded = self.decoder(k)
                    # encoded = encoded[0]
                    temp.append(decoded)

            tp = "".join(temp)
            cap_preds.append(tp)

        return cap_preds

    def forward(self, logits):
        # cap_preds = []
        # for vp in logits:
        current_preds = self.decode_predictions(logits)
        # cap_preds.extends(current_preds)

        return current_preds


def test():
    """Function of test"""
    from torchinfo import summary

    loss_fn = CTCLoss()
    model = CRNN(in_channels=3, n_chars=64)
    summary(model, input_size=(16, N_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))

    x = torch.randn(4, N_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    y = model(x)
    print("y=", y.size())
    print("total=", sum(y[0, 0, :]))

    target = torch.randint(1, 20, (4, 5))
    loss = loss_fn(y, target, torch.tensor([5]*4))
    print("loss=", loss.item())


if __name__ == '__main__':
    test()

