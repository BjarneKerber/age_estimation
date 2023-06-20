import torch
import torchvision
import torch.nn.functional as F
from pathlib import Path


class DeepNormalModel(torch.nn.Module):
    def __init__(self, model_type, n_hidden=256, state=False, in_channels=1, out_channels=1, dropout_rate=0):
        super().__init__()

        self.model_type = model_type
        self.dropout_rate = dropout_rate

        if model_type == "densenet":
            self.extractor = torchvision.models.densenet121(pretrained=False).features

            self.extractor[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7),
                                                stride=(2, 2), padding=(3, 3), bias=False)

            self.hidden = torch.nn.Linear(1024, n_hidden)
            self.linear = torch.nn.Linear(n_hidden, out_channels)
            self.relu = torch.nn.ReLU()

            self.dropout = torch.nn.Dropout(p=dropout_rate)

            if state:
                # get the pretrained
                self.load_state_dict(torch.load(Path(state)))

            # placeholder for the gradients
            self.gradients = None

        elif model_type == "resnet":
            _model = torchvision.models.resnet18(pretrained=True)

            self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = _model.bn1
            self.relu = torch.nn.ReLU(inplace=True)
            self.maxpool = _model.maxpool
            self.layer1 = _model.layer1
            self.layer2 = _model.layer2
            self.layer3 = _model.layer3
            self.layer4 = _model.layer4
            self.avgpool = _model.avgpool
            self.fc = torch.nn.Linear(512, out_channels)

            del _model

            if state:
                # get the pretrained
                self.load_state_dict(torch.load(Path(state)))

            # placeholder for the gradients
            self.gradients = None

        elif model_type == "vgg":
            _model = torchvision.models.vgg11_bn(pretrained=True)
            self.extractor = _model.features[:-1]

            self.extractor[0] = torch.nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

            self.maxpool = _model.features[-1:]

            self.avgpool = _model.avgpool

            self.fc = torch.nn.Sequential(
                torch.nn.Linear(512 * 7 * 7, n_hidden),
                torch.nn.ReLU(True),
                torch.nn.Dropout(p=dropout_rate),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.ReLU(True),
                torch.nn.Dropout(p=dropout_rate),
                torch.nn.Linear(n_hidden, out_channels),
            )

            if state:
                # get the pretrained
                self.load_state_dict(torch.load(Path(state)))

            # placeholder for the gradients
            self.gradients = None

        else:
            print("not implemented")

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        if self.model_type == "densenet":
            # Extract features
            features = self.extractor(x)

            # register the hook
            if features.requires_grad:
                h = features.register_hook(self.activations_hook)

            # Apply ReLU and pooling
            features = F.relu(features, inplace=True)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)

            hidden = self.hidden(features)
            hidden = self.relu(hidden)
            if self.dropout_rate > 0:
                hidden = self.dropout(hidden)
            result = self.linear(hidden)

        elif self.model_type == "resnet":
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            # register the hook
            if x.requires_grad:
                h = x.register_hook(self.activations_hook)

            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            result = self.fc(x)

            return result

        elif self.model_type == "vgg":

            # Extract Features and Average Pool
            features = self.extractor(x)

            # register the hook
            if features.requires_grad:
                h = features.register_hook(self.activations_hook)

            # apply the remaining pooling
            features = self.maxpool(features)
            features = self.avgpool(features)
            features = torch.flatten(features, 1)

            # fully connected layers
            result = self.fc(features)

            return result

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation extraction
    def get_activations(self, x):
        if self.model_type == "vgg":
            return self.extractor(x)
        if self.model_type == "resnet":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            #             x = self.layer4(x)
            return x
