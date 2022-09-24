from tkinter import Y
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from loss import dice_loss




Y_LENGTH = 451
X_LENGTH = 1023
LR=1e-5

''' Res2Conv1d + BatchNorm1d + ReLU
'''

#model 1 ECAPA_TDNN
class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width,
                              kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


''' Conv1d + BatchNorm1d + ReLU
'''


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


''' The SE connection of 1D case.
'''


class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


''' SE-Res2Block.
    Note: residual connection is implemented in the ECAPA_TDNN model, not here.
'''


def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride,
                         padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)
    )


''' Attentive weighted mean and standard deviation pooling.
'''


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        # equals W and b in the paper
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        # equals V and k in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


''' Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".
    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation, 
    because it brings little improvment but significantly increases model parameters. 
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
'''


class ECAPA_TDNN(pl.LightningModule):
    def __init__(self, in_channels=Y_LENGTH, channels=256, embd_dim=X_LENGTH):
        super().__init__()
        self.layer1 = Conv1dReluBn(
            in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)
        self.layer3 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(
            channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536, 128)
        self.bn1 = nn.BatchNorm1d(3072)
        self.linear = nn.Linear(3072, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)
        self.save_hyperparameters()

    def forward(self, x):
        
        x= x.transpose(1,2) #[b,X,Y]d by hsz 

        # source code
        x = x.transpose(1, 2)
        out1 = self.layer1(x)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        out = self.bn2(self.linear(out))
        return out

    def loss_fn(self, out, target):
        return nn.L1Loss()(out, target)

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"],batch["y"]

        out = self(x)
        loss = self.loss_fn(out, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y =batch["x"],batch["y"]

        out = self(x)
        loss = self.loss_fn(out, y)
        self.log('val_loss', loss)
        return loss

#model 2 TDNN
class TDNN(pl.LightningModule):

    def __init__(
        self,
        input_dim=Y_LENGTH,
        output_dim=1,
        context_size=7,
        stride=1,
        dilation=1,
        batch_norm=True,
        dropout_p=0.0
    ):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_dim*context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        self.save_hyperparameters()

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        x=x.transpose(1,2)# by hsz [B,X,Y]
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(
            self.input_dim, d)
        
       

        x = x.unsqueeze(1)
        
        x=F.pad(x, pad = [0,0, self.context_size//2,self.context_size//2], mode='constant')
        

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(self.dilation, 1)
        )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.nonlinearity(x)

        if self.dropout_p:
            x = self.drop(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        return x

    def loss_fn(self, out, target):
        
        return nn.L1Loss()(out.squeeze(2), target)

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"],batch["y"]

        out = self(x)
        loss = self.loss_fn(out, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y =batch["x"],batch["y"]

        out = self(x)
        loss = self.loss_fn(out, y)
        self.log('val_loss', loss)
        return loss



#model 3 Simple Unet

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class SimpleUNet(pl.LightningModule):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024,512,256, 128, 64), num_class=2, retain_dim=True, out_sz=(Y_LENGTH,X_LENGTH)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz=out_sz

    def forward(self, x):
        #print(x)
        x=x.transpose(1,3).transpose(2,3)#by hsz [B,3,Y,X]
        
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        out =F.softmax(out, dim=1) # add change by hsz[B,num_class,y,x]
        return out 

    def loss_fn(self, out, target):
        #print(out,target,out.shape,target.shape)
        return nn.CrossEntropyLoss()(out, target)

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"],batch["y"]

        out = self(x)
        loss = self.loss_fn(out, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y =batch["x"],batch["y"]

        out = self(x)
        loss = self.loss_fn(out, y)
        self.log('val_loss', loss)
        return loss


#model 4 complete UNet
#https://github.com/milesial/Pytorch-UNet


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(pl.LightningModule):
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x=x.transpose(1,3).transpose(2,3)#by hsz [B,3,Y,X]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        #out=F.sigmoid(x.squeeze(1)) # hsz  if out is [B,1,Y,X]
        #out =F.softmax(out, dim=1) #hsz if out is [B,2,Y,X]
        return out

    def loss_fn(self, out, target):
        #print(out,target,out.shape,target.shape)
        #Pytorch中, CrossEntropyLoss是包含了softmax的内容的，我们损失函数使用了CrossEntropyLoss, 那么网络的最后一层就不用softmax
        loss=nn.CrossEntropyLoss()(out, target)  + dice_loss(F.softmax(out, dim=1).float(),
                                       F.one_hot(target, 2).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
        return loss

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"],batch["y"]

        out = self(x)
        loss = self.loss_fn(out, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y =batch["x"],batch["y"]

        out = self(x)
        loss = self.loss_fn(out, y)
        self.log('val_loss', loss)
        return loss
