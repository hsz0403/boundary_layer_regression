from tkinter import Y
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from loss import FocalLoss, dice_loss




Y_LENGTH = 451
X_LENGTH = 1023
LR=1e-5

''' Res2Conv1d + BatchNorm1d + ReLU
'''

#model 1 Simple Unet

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


#model 2 complete UNet
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
    def __init__(self, n_channels=5, n_classes=1, bilinear=True):
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
        #x=x.transpose(1,3).transpose(2,3)#by hsz [B,3+2,Y,X]
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
        out =F.softmax(out.squeeze(1), dim=1) 
        return out

    def loss_fn(self, out, target):
        #print(out,target,out.shape,target.shape)
        #Pytorch中, CrossEntropyLoss是包含了softmax的内容的，我们损失函数使用了CrossEntropyLoss, 那么网络的最后一层就不用softmax
        #loss=1*nn.CrossEntropyLoss()(out, target)  + 0*dice_loss(F.softmax(out, dim=1).float(),F.one_hot(target, 2).permute(0, 3, 1, 2).float(),multiclass=True)+0*FocalLoss()(out, target)
        
        
        loss=nn.NLLLoss()(torch.log(out), target)
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

    
    #model 3 idea by prof.Zhao segnet(?)
    

class SegNet(pl.LightningModule):

    def __init__(self, BN_momentum=0.5):
        super(SegNet, self).__init__()

		#SegNet Architecture
		#Takes input of size in_chn = 3 (RGB images have 3 channels)
		#Outputs size label_chn (N # of classes)

		#ENCODING consists of 5 stages
		#Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
		#Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

		#General Max Pool 2D for ENCODING layers
		#Pooling indices are stored for Upsampling in DECODING layers

        self.in_chn = 5
        self.out_chn = 1

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)


		#DECODING consists of 5 stages
		#Each stage corresponds to their respective counterparts in ENCODING

		#General Max Pool 2D/Upsampling for DECODING layers
        self.MaxDe = nn.MaxUnpool2d(2, stride=2) 

        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)
		
        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
		
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
		
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

		
        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

    def forward(self, x):

		#ENCODE LAYERS
		#Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x))) 
		
        x = F.relu(self.BNEn12(self.ConvEn12(x))) 
		
        x, ind1 = self.MaxEn(x)
		
        size1 = x.size()

		#Stage 2
		
        x = F.relu(self.BNEn21(self.ConvEn21(x))) 
		
        x = F.relu(self.BNEn22(self.ConvEn22(x))) 
		
        x, ind2 = self.MaxEn(x)
		
        size2 = x.size()

		#Stage 3
		
        x = F.relu(self.BNEn31(self.ConvEn31(x))) 
		
        x = F.relu(self.BNEn32(self.ConvEn32(x))) 
		
        x = F.relu(self.BNEn33(self.ConvEn33(x))) 	
		
        x, ind3 = self.MaxEn(x)
		
        size3 = x.size()

		#Stage 4
		
        x = F.relu(self.BNEn41(self.ConvEn41(x))) 
		
        x = F.relu(self.BNEn42(self.ConvEn42(x))) 
		
        x = F.relu(self.BNEn43(self.ConvEn43(x))) 	
		
        x, ind4 = self.MaxEn(x)
		
        size4 = x.size()

		#Stage 5
		
        x = F.relu(self.BNEn51(self.ConvEn51(x))) 
		
        x = F.relu(self.BNEn52(self.ConvEn52(x))) 
		
        x = F.relu(self.BNEn53(self.ConvEn53(x))) 	
		
        x, ind5 = self.MaxEn(x)
		
        size5 = x.size()


		#DECODE LAYERS
		#Stage 5

		
        x = self.MaxDe(x, ind5, output_size=size4)

		
        x = F.relu(self.BNDe53(self.ConvDe53(x)))

		
        x = F.relu(self.BNDe52(self.ConvDe52(x)))

		
        x = F.relu(self.BNDe51(self.ConvDe51(x)))


		#Stage 4
		
        x = self.MaxDe(x, ind4, output_size=size3)
		
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
		
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
		
        x = F.relu(self.BNDe41(self.ConvDe41(x)))


		#Stage 3
		
        x = self.MaxDe(x, ind3, output_size=size2)
		
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
		
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
		
        x = F.relu(self.BNDe31(self.ConvDe31(x)))


		#Stage 2
		
        x = self.MaxDe(x, ind2, output_size=size1)
		
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
		
        x = F.relu(self.BNDe21(self.ConvDe21(x)))


		#Stage 1
		
        x = self.MaxDe(x, ind1)
		
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
		
        
        x = self.ConvDe11(x)


        x =F.softmax(x.squeeze(1), dim=1) 
        return x
    def loss_fn(self, out, target):
        #print(out,target,out.shape,target.shape)
        #Pytorch中, CrossEntropyLoss是包含了softmax的内容的，我们损失函数使用了CrossEntropyLoss, 那么网络的最后一层就不用softmax
        #loss=1*nn.CrossEntropyLoss()(out, target)  + 0*dice_loss(F.softmax(out, dim=1).float(),F.one_hot(target, 2).permute(0, 3, 1, 2).float(),multiclass=True)+0*FocalLoss()(out, target)
        
        
        loss=nn.NLLLoss()(torch.log(out), target)
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

    
