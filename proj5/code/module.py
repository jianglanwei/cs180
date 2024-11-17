import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchsummary
from animator import animator

class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 
                                       kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.AvgPool2d(7)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.gelu(x)
        return x
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class Unflatten(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.unflatten = nn.ConvTranspose2d(in_channels, in_channels, 
                                            kernel_size=7, stride=7, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unflatten(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels)
        self.conv2 = Conv(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downconv = DownConv(in_channels, out_channels)
        self.convblock = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downconv(x)
        x = self.convblock(x)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upconv = UpConv(in_channels, out_channels)
        self.convblock = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        x = self.convblock(x)
        return x
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

class UnconditionalUNet(nn.Module):
    """unconditional unet module."""
    def __init__(self, in_channels: int, num_hiddens: int):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=in_channels, 
                                    out_channels=num_hiddens)
        self.convblock2 = ConvBlock(in_channels=2*num_hiddens, 
                                    out_channels=num_hiddens)
        self.downblock1 = DownBlock(in_channels=num_hiddens,
                                    out_channels=num_hiddens)
        self.downblock2 = DownBlock(in_channels=num_hiddens,
                                    out_channels=2*num_hiddens)
        self.flatten = Flatten()
        self.unflatten = Unflatten(in_channels=2*num_hiddens)
        self.upblock1 = UpBlock(in_channels=2*num_hiddens,
                                out_channels=num_hiddens)
        self.upblock2 = UpBlock(in_channels=4*num_hiddens,
                                out_channels=num_hiddens)
        self.conv = nn.Conv2d(in_channels=num_hiddens, out_channels=1,
                              kernel_size=3, stride=1, padding=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-3:] == (1, 28, 28), "x shape should be (N, 1, 28, 28)."
        in1 = self.convblock1(x)
        in2 = self.downblock1(in1)
        in3 = self.downblock2(in2)
        flat = self.flatten(in3)
        out3 = self.unflatten(flat)
        cat3 = torch.cat((in3, out3), dim=1)
        out2 = self.upblock2(cat3)
        cat2 = torch.cat((in2, out2), dim=1)
        out1 = self.upblock1(cat2)
        cat1 = torch.cat((in1, out1), dim=1)
        out0 = self.convblock2(cat1)
        out = self.conv(out0)
        return out
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

class FCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class TimeConditionalUNet(nn.Module):
    """time conditional unet module."""
    def __init__(self, in_channels: int, num_hiddens: int):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=in_channels, 
                                    out_channels=num_hiddens)
        self.convblock2 = ConvBlock(in_channels=2*num_hiddens, 
                                    out_channels=num_hiddens)
        self.downblock1 = DownBlock(in_channels=num_hiddens,
                                    out_channels=num_hiddens)
        self.downblock2 = DownBlock(in_channels=num_hiddens,
                                    out_channels=2*num_hiddens)
        self.flatten = Flatten()
        self.unflatten = Unflatten(in_channels=2*num_hiddens)
        self.upblock1 = UpBlock(in_channels=2*num_hiddens,
                                out_channels=num_hiddens)
        self.upblock2 = UpBlock(in_channels=4*num_hiddens,
                                out_channels=num_hiddens)
        self.fcblock2 = FCBlock(in_channels=1, out_channels=num_hiddens)
        self.fcblock3 = FCBlock(in_channels=1, out_channels=2*num_hiddens)
        self.conv = nn.Conv2d(in_channels=num_hiddens, out_channels=1,
                              kernel_size=3, stride=1, padding=1)


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert x.shape[-3:] == (1, 28, 28), "x shape should be (N, 1, 28, 28)."
        assert t.shape[-3:] == (1, 1, 1), "t shape should be (N, 1, 1, 1)."
        assert x.shape[0] == t.shape[0], "incompatible N."
        in1 = self.convblock1(x)
        in2 = self.downblock1(in1)
        in3 = self.downblock2(in2)
        flat = self.flatten(in3)
        out3 = self.unflatten(flat)
        add3 = self.fcblock3(t).reshape(t.shape[0], -1, 1, 1)
        out3 = out3 + add3
        cat3 = torch.cat((in3, out3), dim=1)
        out2 = self.upblock2(cat3)
        add2 = self.fcblock2(t).reshape(t.shape[0], -1, 1, 1)
        out2 = out2 + add2
        cat2 = torch.cat((in2, out2), dim=1)
        out1 = self.upblock1(cat2)
        cat1 = torch.cat((in1, out1), dim=1)
        out0 = self.convblock2(cat1)
        out = self.conv(out0)
        return out
    
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward(x, t)


class ClassConditionalUNet(nn.Module):
    """class conditional unet module."""
    def __init__(self, in_channels: int, num_hiddens: int):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=in_channels, 
                                    out_channels=num_hiddens)
        self.convblock2 = ConvBlock(in_channels=2*num_hiddens, 
                                    out_channels=num_hiddens)
        self.downblock1 = DownBlock(in_channels=num_hiddens,
                                    out_channels=num_hiddens)
        self.downblock2 = DownBlock(in_channels=num_hiddens,
                                    out_channels=2*num_hiddens)
        self.flatten = Flatten()
        self.unflatten = Unflatten(in_channels=2*num_hiddens)
        self.upblock1 = UpBlock(in_channels=2*num_hiddens,
                                out_channels=num_hiddens)
        self.upblock2 = UpBlock(in_channels=4*num_hiddens,
                                out_channels=num_hiddens)
        self.fcblock2_time = FCBlock(in_channels=1, out_channels=num_hiddens)
        self.fcblock3_time = FCBlock(in_channels=1, out_channels=2*num_hiddens)
        self.fcblock2_class = FCBlock(in_channels=10, out_channels=num_hiddens)
        self.fcblock3_class = FCBlock(in_channels=10, out_channels=2*num_hiddens)
        self.conv = nn.Conv2d(in_channels=num_hiddens, out_channels=1,
                              kernel_size=3, stride=1, padding=1)


    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert x.shape[-3:] == (1, 28, 28), "x shape should be (N, 1, 28, 28)."
        assert t.shape[-3:] == (1, 1, 1), "t shape should be (N, 1, 1, 1)."
        assert c.shape[-3:] == (1, 1, 10), "c shape should be (N, 1, 1, 10)."
        assert x.shape[0] == t.shape[0] and x.shape[0] == c.shape[0], "incompatible N."
        in1 = self.convblock1(x)
        in2 = self.downblock1(in1)
        in3 = self.downblock2(in2)
        flat = self.flatten(in3)
        out3 = self.unflatten(flat)
        add3 = self.fcblock3_time(t).reshape(x.shape[0], -1, 1, 1)
        mul3 = self.fcblock3_class(c).reshape(x.shape[0], -1, 1, 1)
        out3 = out3 * mul3 + add3
        cat3 = torch.cat((in3, out3), dim=1)
        out2 = self.upblock2(cat3)
        add2 = self.fcblock2_time(t).reshape(x.shape[0], -1, 1, 1)
        mul2 = self.fcblock2_class(c).reshape(x.shape[0], -1, 1, 1)
        out2 = out2 * mul2 + add2
        cat2 = torch.cat((in2, out2), dim=1)
        out1 = self.upblock1(cat2)
        cat1 = torch.cat((in1, out1), dim=1)
        out0 = self.convblock2(cat1)
        out = self.conv(out0)
        return out
    
    def __call__(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.forward(x, t, c)
