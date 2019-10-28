import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class EncodingLayer(nn.Module):
    def __init__(self, input_channel_count, output_channel_count, use_instance_norm, use_activation=True):
        super(EncodingLayer, self).__init__()
        
        self.input_channel_count  = input_channel_count
        self.output_channel_count = output_channel_count
        self.use_instance_norm    = use_instance_norm
        self.use_activation       = use_activation

        self.conv       = nn.Conv2d(input_channel_count, output_channel_count, (4, 4), stride=2, padding=(1,1))
        self.norm       = torch.nn.InstanceNorm2d(output_channel_count, 1e-5, affine=True) if use_instance_norm else None
        self.leaky_relu = nn.LeakyReLU(0.2) if use_activation else None

    def forward(self, x):
        if self.leaky_relu is not None:
            x = self.leaky_relu(x)

        x = self.conv(x)

        if self.norm is not None:
            x = self.norm(x)
        
        return x

class DecodingLayer(nn.Module):
    def __init__(self, input_channel_count, output_channel_count, use_instance_norm, use_dropout, use_activation=True):
        super(DecodingLayer, self).__init__()
        
        self.input_channel_count  = input_channel_count
        self.output_channel_count = output_channel_count
        self.use_instance_norm    = use_instance_norm
        self.use_dropout          = use_dropout
        self.use_activation       = use_activation

        self.deconv     = nn.ConvTranspose2d(input_channel_count, output_channel_count, (4, 4), stride=2, padding=(1, 1))
        self.norm       = torch.nn.InstanceNorm2d(output_channel_count, 1e-5, affine=True) if use_instance_norm else None
        self.dropout    = nn.Dropout(0.5) if use_dropout else None
        self.leaky_relu = nn.LeakyReLU(0.2) if use_activation else None

    def forward(self, x):
        if self.leaky_relu is not None:
            x = self.leaky_relu(x)

        x = self.deconv(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class Generator(nn.Module):
    input_channel_count  = 3
    output_channel_count = 64 
    filters_count        = 64 # "ngf" in the original code

    def __init__(self, output_channel_count, number_of_filters = 64):
        super(Generator, self).__init__()
        self.number_of_filters    = number_of_filters
        self.output_channel_count = output_channel_count

        self.create_layers()

    def create_layers(self):
        # TODO: Handle useCoordConv

        # TODO: Handle global track   

        self.enc1 = EncodingLayer(self.input_channel_count,       self.number_of_filters    , False, False) # encoder_1: [batch, 256, 256, 3      ] => [batch, 128, 128, ngf    ]
        self.enc2 = EncodingLayer(self.enc1.output_channel_count, self.number_of_filters * 2,  True)        # encoder_2: [batch, 128, 128, ngf    ] => [batch,  64,  64, ngf * 2]
        self.enc3 = EncodingLayer(self.enc2.output_channel_count, self.number_of_filters * 4,  True)        # encoder_3: [batch,  64,  64, ngf * 2] => [batch,  32,  32, ngf * 4]
        self.enc4 = EncodingLayer(self.enc3.output_channel_count, self.number_of_filters * 8,  True)        # encoder_4: [batch,  32,  32, ngf * 4] => [batch,  16,  16, ngf * 8]
        self.enc5 = EncodingLayer(self.enc4.output_channel_count, self.number_of_filters * 8,  True)        # encoder_5: [batch,  16,  16, ngf * 8] => [batch,   8,   8, ngf * 8]
        self.enc6 = EncodingLayer(self.enc5.output_channel_count, self.number_of_filters * 8,  True)        # encoder_6: [batch,   8,   8, ngf * 8] => [batch,   4,   4, ngf * 8]
        self.enc7 = EncodingLayer(self.enc6.output_channel_count, self.number_of_filters * 8,  True)        # encoder_7: [batch,   4,   4, ngf * 8] => [batch,   2,   2, ngf * 8]
        self.enc8 = EncodingLayer(self.enc6.output_channel_count, self.number_of_filters * 8, False)        # encoder_8: [batch,   2,   2, ngf * 8] => [batch,   1,   1, ngf * 8]                  

        self.dec8 = DecodingLayer(self.number_of_filters * 8,         self.number_of_filters * 8,  True,  True) # decoder_8: [batch,  1,  1,      ngf * 8] => [batch,   2,   2, ngf * 8]
        self.dec7 = DecodingLayer(2 * self.dec8.output_channel_count, self.number_of_filters * 8,  True,  True) # decoder_7: [batch,  2,  2, 2 * ngf * 8 ] => [batch,   4,   4, ngf * 8]
        self.dec6 = DecodingLayer(2 * self.dec7.output_channel_count, self.number_of_filters * 8,  True,  True) # decoder_6: [batch,  4,  4, 2 * ngf * 8 ] => [batch,   8,   8, ngf * 8] 
        self.dec5 = DecodingLayer(2 * self.dec6.output_channel_count, self.number_of_filters * 8,  True, False) # decoder_5: [batch,  8,  8, 2 * ngf * 8 ] => [batch,  16,  16, ngf * 8]
        self.dec4 = DecodingLayer(2 * self.dec5.output_channel_count, self.number_of_filters * 4,  True, False) # decoder_4: [batch, 16, 16, 2 * ngf * 8 ] => [batch,  32,  32, ngf * 4]
        self.dec3 = DecodingLayer(2 * self.dec4.output_channel_count, self.number_of_filters * 2,  True, False) # decoder_3: [batch, 32, 32, 2 * ngf * 4 ] => [batch,  64,  64, ngf * 2]
        self.dec2 = DecodingLayer(2 * self.dec3.output_channel_count, self.number_of_filters    ,  True, False) # decoder_2: [batch, 64, 64, 2 * ngf * 2 ] => [batch, 128, 128, ngf    ]
        self.dec1 = DecodingLayer(2 * self.dec2.output_channel_count, self.output_channel_count , False, False) # decoder_1: [batch, 64, 64, 2 * ngf     ] => [batch, 128, 128, 64     ]                   

    def forward(self, input):
        # Encoding
        down1 = self.enc1(input)
        down2 = self.enc2(down1)
        down3 = self.enc3(down2)
        down4 = self.enc4(down3)
        down5 = self.enc5(down4)
        down6 = self.enc6(down5)
        down7 = self.enc7(down6)
        down8 = self.enc8(down7)

        # Decoding
        up8 = self.dec8(                down8)
        up7 = self.dec7(torch.cat((up8, down7), dim=1))
        up6 = self.dec6(torch.cat((up7, down6), dim=1))
        up5 = self.dec5(torch.cat((up6, down5), dim=1))
        up4 = self.dec4(torch.cat((up5, down4), dim=1))
        up3 = self.dec3(torch.cat((up4, down3), dim=1))
        up2 = self.dec2(torch.cat((up3, down2), dim=1))
        up1 = self.dec1(torch.cat((up2, down1), dim=1))

        return up1

def gamma_decode(images):
    return torch.pow(images, 2.2)

def gamma_encode(images):
    return torch.pow(images, 1.0/2.2)

writer = SummaryWriter("./logs")

generator = Generator(3).cuda()
print(generator)

img1 = plt.imread("./out1.png")
img2 = plt.imread("./out2.png")
img3 = plt.imread("./out3.png")
input = gamma_decode(torch.tensor([img1, img2]).permute(0, 3, 1, 2).cuda())
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
for epoch in range(50):

    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = generator(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()    # Does the update

    writer.add_scalar("loss", loss.item(), epoch)

    print("Epoch {:d}, loss: {:f}".format(epoch + 1, loss.item()))

writer.add_graph(generator, input)
writer.close()

plt.imshow(gamma_encode(generator(input)[0].clone().cpu().detach().permute(1, 2, 0)))
plt.waitforbuttonpress()

plt.imshow(gamma_encode(generator(input)[1].clone().cpu().detach().permute(1, 2, 0)))
plt.waitforbuttonpress()

plt.imshow(gamma_encode(generator(gamma_decode(torch.tensor([img3]).permute(0, 3, 1, 2).cuda()))[0].clone().cpu().detach().permute(1, 2, 0)))
plt.waitforbuttonpress()

