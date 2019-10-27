import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Generator(nn.Module):
    input_channel_count  = 3
    output_channel_count = 3 
    filters_count        = 64 # "ngf" in the original code
    conv_kernel_size     = (4, 4)

    def __init__(self, number_of_filters, generator_output_channels):
        super(Generator, self).__init__()
        self.number_of_filters         = number_of_filters
        self.generator_output_channels = generator_output_channels

        self.create_encoder()
        self.create_decoder()
    
    def create_conv2d(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, self.conv_kernel_size, stride=2, padding=(1,1))

    def create_deconv2d(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, self.conv_kernel_size, stride=2, padding=(1, 1))

    def create_encoder(self):
        # TODO: Handle useCoordConv

        # TODO: Handle global track

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv1 = self.create_conv2d(self.input_channel_count  , self.number_of_filters    ) # encoder_1: [batch, 256, 256, 3      ] => [batch, 128, 128, ngf    ]
        self.conv2 = self.create_conv2d(self.number_of_filters    , self.number_of_filters * 2) # encoder_2: [batch, 128, 128, ngf    ] => [batch,  64,  64, ngf * 2]
        self.conv3 = self.create_conv2d(self.number_of_filters * 2, self.number_of_filters * 4) # encoder_3: [batch,  64,  64, ngf * 2] => [batch,  32,  32, ngf * 4]
        self.conv4 = self.create_conv2d(self.number_of_filters * 4, self.number_of_filters * 8) # encoder_4: [batch,  32,  32, ngf * 4] => [batch,  16,  16, ngf * 8]
        self.conv5 = self.create_conv2d(self.number_of_filters * 8, self.number_of_filters * 8) # encoder_5: [batch,  16,  16, ngf * 8] => [batch,   8,   8, ngf * 8]
        self.conv6 = self.create_conv2d(self.number_of_filters * 8, self.number_of_filters * 8) # encoder_6: [batch,   8,   8, ngf * 8] => [batch,   4,   4, ngf * 8]
        self.conv7 = self.create_conv2d(self.number_of_filters * 8, self.number_of_filters * 8) # encoder_7: [batch,   4,   4, ngf * 8] => [batch,   2,   2, ngf * 8]
        self.conv8 = self.create_conv2d(self.number_of_filters * 8, self.number_of_filters * 8) # encoder_8: [batch,   2,   2, ngf * 8] => [batch,   1,   1, ngf * 8]
        self.conv9 = self.create_conv2d(self.number_of_filters * 8, self.number_of_filters * 8)

    def create_decoder(self):
        self.dropout = nn.Dropout(0.5)

        self.deconv8 = self.create_deconv2d(self.number_of_filters * 8,     self.number_of_filters * 8) # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8]
        self.deconv7 = self.create_deconv2d(2 * self.number_of_filters * 8, self.number_of_filters * 8) # decoder_7: [batch, 2, 2, 2 * ngf * 8 ] => [batch, 4, 4, ngf * 8]
        self.deconv6 = self.create_deconv2d(2 * self.number_of_filters * 8, self.number_of_filters * 8) # decoder_6: [batch, 4, 4, 2 * ngf * 8 ] => [batch, 8, 8, ngf * 8] 
        self.deconv5 = self.create_deconv2d(2 * self.number_of_filters * 8, self.number_of_filters * 8) # decoder_5: [batch, 8, 8, 2 * ngf * 8 ] => [batch, 16, 16, ngf * 8
        self.deconv4 = self.create_deconv2d(2 * self.number_of_filters * 8, self.number_of_filters * 4) # decoder_4: [batch, 16, 16, 2 * ngf * 8 ] => [batch, 32, 32, ngf * 4]
        self.deconv3 = self.create_deconv2d(2 * self.number_of_filters * 4, self.number_of_filters * 2) # decoder_3: [batch, 32, 32, ngf * 4] => [batch, 64, 64, ngf * 2]
        self.deconv2 = self.create_deconv2d(2 * self.number_of_filters * 2, self.number_of_filters    ) # decoder_2: [batch, 64, 64, ngf * 2] => [batch, 128, 128, ngf]
        self.deconv1 = self.create_deconv2d(2 * self.number_of_filters    , self.output_channel_count ) # decoder_2: [batch, 64, 64, ngf * 2] => [batch, 128, 128, 64]

    def forward(self, input):
        down1 = self.conv1(input)
        down2 = self.conv2(self.leaky_relu(down1))
        down3 = self.conv3(self.leaky_relu(down2))
        down4 = self.conv4(self.leaky_relu(down3))
        down5 = self.conv5(self.leaky_relu(down4))
        down6 = self.conv6(self.leaky_relu(down5))
        down7 = self.conv7(self.leaky_relu(down6))
        down8 = self.conv8(self.leaky_relu(down7))

        up8 = self.dropout(self.deconv8(self.leaky_relu(down8)))
        up7 = self.dropout(self.deconv7(self.leaky_relu(torch.cat((up8, down7), dim=1))))
        up6 = self.dropout(self.deconv6(self.leaky_relu(torch.cat((up7, down6), dim=1))))
        up5 =              self.deconv5(self.leaky_relu(torch.cat((up6, down5), dim=1)))
        up4 =              self.deconv4(self.leaky_relu(torch.cat((up5, down4), dim=1)))
        up3 =              self.deconv3(self.leaky_relu(torch.cat((up4, down3), dim=1)))
        up2 =              self.deconv2(self.leaky_relu(torch.cat((up3, down2), dim=1)))
        up1 =              self.deconv1(self.leaky_relu(torch.cat((up2, down1), dim=1)))

        return up1

def gamma_decode(images):
    return torch.pow(images, 2.2)

def gamma_encode(images):
    return torch.pow(images, 1.0/2.2)

generator = Generator(64, 64).cuda()
print(generator)

img1 = plt.imread("./out1.png")
img2 = plt.imread("./out2.png")
img3 = plt.imread("./out3.png")
input = gamma_decode(torch.tensor([img1, img2]).permute(0, 3, 1, 2).cuda())
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)

for epoch in range(100):

    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = generator(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()    # Does the update

    print("Epoch {:d}, loss: {:f}".format(epoch + 1, loss.item()))

plt.imshow(gamma_encode(generator(input)[0].clone().cpu().detach().permute(1, 2, 0)))
plt.waitforbuttonpress()

plt.imshow(gamma_encode(generator(input)[1].clone().cpu().detach().permute(1, 2, 0)))
plt.waitforbuttonpress()

plt.imshow(gamma_encode(generator(torch.tensor([img3]).permute(0, 3, 1, 2).cuda())[0].clone().cpu().detach().permute(1, 2, 0)))
plt.waitforbuttonpress()