import torch
import torch.nn as  nn
import torch.nn.functional as F
import functools
import config as c
from utils import quantization, cat_map


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid): # nn.Sigmoid
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, output_function=output_function)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if output_function == nn.Tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class RevealNet(nn.Module):
    def __init__(self, ic=3, oc=3, nhf=64, output_function=nn.Sigmoid):
        super(RevealNet, self).__init__()
        # input is (3) x 256 x 256
        self.main = nn.Sequential(
            nn.Conv2d(ic, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf*4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, oc, 3, 1, 1),
            output_function()
        )

    def forward(self, input):
        output=self.main(input)
        return output


def quantization_pert(tensor):
    return (torch.round(tensor*255+127)-127)/255  # tensor+0.5 ~ [0.5-eps, 0.5+eps]

class Model(nn.Module):
    
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.encoder = UnetGenerator(input_nc=3, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
        self.decoder = RevealNet()
    
    def forward(self, secret, cover):
        if c.mode=='train':
            stego_resi = c.cids_eps * (2*self.encoder(secret)-1)   # [-0.2~0.2]   
            quantization_noise = (quantization_pert(stego_resi) - stego_resi).detach()
            stego_resi = stego_resi + quantization_noise
            stego = cover + stego_resi
            secret_rev = self.decoder(stego_resi)
            return stego, stego_resi, secret_rev
        else: # test
            # embedding for sender
            stego_resi = quantization_pert(c.cids_eps * (2*self.encoder(secret)-1))   # [-0.2~0.2] 
            if c.obfuscate_secret_image == False:
                obfuscated_resi = cat_map(stego_resi.cpu(), obfuscate=True).cuda()
            else:
                obfuscated_resi = stego_resi
            stego = cover + obfuscated_resi
            stego = torch.clip(stego, 0, 1)

            # send the stego to receiver through public channel.
            
            # extraction for receiver
            obfuscated_resi = stego - cover
            if c.obfuscate_secret_image == False:
                stego_resi = cat_map(obfuscated_resi.cpu(), obfuscate=False).cuda()
            else:
                stego_resi = obfuscated_resi
            secret_rev = quantization(self.decoder(stego_resi))
            stego_resi = obfuscated_resi

            return stego, obfuscated_resi, stego_resi, secret_rev


