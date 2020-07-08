import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, Blocktype = 'simple'):
        super(BasicBlock, self).__init__()
        # each basic block has some specific layers and a shortcut connection
        # self.in_chan, self.out_chan = in_channels, out_channels 
        self.directshortcut = (in_channels == out_channels)
        self.stride, self.padding =0, 0
        if(self.directshortcut == True):
            self.stride, self.padding = 1, 1
        else:
            self.stride, self.padding = 2, 1
        
        # print(self.stride, self.padding)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = self.stride, padding = self.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 2),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        scut = x
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        if(self.directshortcut == False):
            scut = self.shortcut(scut)
        return out + scut



class Block(nn.Module):
    def __init__(self, in_channels, out_channels, Block_length):
        super(Block, self).__init__()
        # each block is made of a a number of basic blocks
        self.basicblocks = nn.ModuleList([BasicBlock(in_channels = in_channels, out_channels = out_channels, Blocktype = 'simple'), *[BasicBlock(in_channels = out_channels, out_channels = out_channels, Blocktype = 'simple') for i in range(Block_length-1)]])

    def forward(self, x):
        out = x
        for basicblock in self.basicblocks:
            out = basicblock(out)
        return(out)

class Encoder(nn.Module):
    def __init__(self, inchannel_size, channel_sizes, Block_lengths):
        super(Encoder, self).__init__()
        blocks_inout_channel_size  = [(inchannel_size, channel_sizes[0]), *[(i, j) for i,j in zip(channel_sizes, channel_sizes[1:])]]
        self.blocks = nn.ModuleList([Block(in_channels = in_chan, out_channels = out_chan, Block_length = Block_lengths[n]) for (in_chan, out_chan), n in zip(blocks_inout_channel_size, Block_lengths)])

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class Resnet(nn.Module):
    def __init__(self, inputsize = 224, channel_sizes = [64, 128, 256, 512], Block_lengths = [2,2,2,2], n_classes = 1000):
        super(Resnet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = channel_sizes[0], kernel_size = 7, stride = 2, padding = 3),#padding =3 for making final size = 112, check bias
            nn.BatchNorm2d(channel_sizes[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
            # outputsize = 56*56
        )
        self.encoder = Encoder(inchannel_size = channel_sizes[0], channel_sizes = channel_sizes, Block_lengths = Block_lengths)
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))        
        self.fc1 = nn.Linear(512, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.encoder(out)
        out = self.globalavgpool(out)
        # print('After avgpool' + str(out.shape))
        out = torch.flatten(out, 1)
        out = self.fc1(out)

        return out


from torchsummary import summary
model = Resnet()
summary(model(Block_lengths=[2,2,2,2], n_classes=1000), input_size = (3, 224, 224))


