import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

class EncodingLayer(nn.Module):
    def __init__(self, input_channel_count, output_channel_count, use_instance_norm, use_activation=True):
        super(EncodingLayer, self).__init__()
        
        self.input_channel_count  = input_channel_count
        self.output_channel_count = output_channel_count
        self.use_instance_norm    = use_instance_norm
        self.use_activation       = use_activation

        self.conv            = nn.Conv2d(input_channel_count, output_channel_count, (4, 4), stride=2, padding=(1,1))
        self.norm            = torch.nn.InstanceNorm2d(output_channel_count, 1e-5, affine=True) if use_instance_norm else None
        self.leaky_relu      = nn.LeakyReLU(0.2) if use_activation else None
        self.fully_connected = nn.Linear(self.output_channel_count, self.output_channel_count, False)

    def forward(self, x, global_track):
        if self.leaky_relu is not None:
            x = self.leaky_relu(x)

        x = self.conv(x)

        mean = torch.mean(x, dim=(2,3), keepdim=False)

        if self.norm is not None:
            x = self.norm(x)
        
        if global_track is not None:
            global_track = self.fully_connected(global_track)
            x = torch.add(x, global_track.unsqueeze(-1).unsqueeze(-1))

        return x, mean

class DecodingLayer(nn.Module):
    def __init__(self, input_channel_count, output_channel_count, use_instance_norm, use_dropout, use_activation=True):
        super(DecodingLayer, self).__init__()
        
        self.input_channel_count  = input_channel_count
        self.output_channel_count = output_channel_count
        self.use_instance_norm    = use_instance_norm
        self.use_dropout          = use_dropout
        self.use_activation       = use_activation

        self.deconv     = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2.0),
            nn.Conv2d(input_channel_count,  output_channel_count, (3, 3), padding=(1, 1)), # TODO: padding_mode="same"
            nn.Conv2d(output_channel_count, output_channel_count, (3, 3), padding=(1, 1)), # TODO: padding_mode="same"
        )
        
        self.norm       = torch.nn.InstanceNorm2d(output_channel_count, 1e-5, affine=True) if use_instance_norm else None
        self.dropout    = nn.Dropout(0.5) if use_dropout else None
        self.leaky_relu = nn.LeakyReLU(0.2) if use_activation else None
        self.fully_connected = nn.Linear(self.output_channel_count, self.output_channel_count, False)

    def forward(self, x, skip_connected_tensor, global_track):
        if skip_connected_tensor is not None:
            x = torch.cat((x, skip_connected_tensor), dim=1)

        if self.leaky_relu is not None:
            x = self.leaky_relu(x)

        x = self.deconv(x)

        mean = torch.mean(x, dim=(2,3), keepdim=False)

        if self.norm is not None:
            x = self.norm(x)

        if global_track is not None:
            global_track = self.fully_connected(global_track)
            x = torch.add(x, global_track.unsqueeze(-1).unsqueeze(-1))

        if self.dropout is not None:
            x = self.dropout(x)
        
        return x, mean

class CoordLayer(nn.Module):
    def __init__(self):
        super(CoordLayer, self).__init__()

    def forward(self, x):
        w          = x.shape[-1] # x.shape = (batch, channels, height, width)
        h          = x.shape[-2] #
        batch_size = x.shape[0]  #

        xcoords_row  = torch.linspace(-1, 1, w).cuda()
        xcoords      = xcoords_row.unsqueeze(0).expand(h, w).unsqueeze(0)
        ycoords      = -1 * torch.transpose(xcoords, dim0=1, dim1=2)
        coords       = torch.cat((xcoords, ycoords), dim=0)
        batch_coords = coords.unsqueeze(0).expand(batch_size, 2, h, w)

        return torch.cat((x, batch_coords), dim=1) # Concatenation in feature dimension

class GlobalTrackLayer(nn.Module):
    def __init__(self, input_channel_count, output_channel_count):
        super(GlobalTrackLayer, self).__init__()

        self.input_channel_count = input_channel_count
        self.output_channel_count = output_channel_count

        self.fully_connected = torch.nn.Linear(input_channel_count, output_channel_count)
        self.selu            = torch.nn.SELU()

    def forward(self, local_mean, global_track):
        if global_track is not None:
            global_track = torch.cat((global_track, local_mean), dim=1)
        else:
            global_track = local_mean

        return self.selu(self.fully_connected(global_track))

class Generator(nn.Module):
    input_channel_count  = 3
    output_channel_count = 64 
    filters_count        = 64 # "ngf" in the original code
    use_coords           = False
    use_global_track     = True

    def __init__(self, output_channel_count, number_of_filters = 64):
        super(Generator, self).__init__()
        self.number_of_filters    = number_of_filters
        self.output_channel_count = output_channel_count

        self.create_layers()

    def create_layers(self):
        self.coord = CoordLayer() if self.use_coords else None   

        encoding_input_channel_count = self.input_channel_count + 2 if self.use_coords else self.input_channel_count
        self.enc1 = EncodingLayer(encoding_input_channel_count,   self.number_of_filters    , False, False) # encoder_1: [batch, 256, 256, 3      ] => [batch, 128, 128, ngf    ]
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

        def bi_noop(x, y):
            return None

        self.gte1 = GlobalTrackLayer(encoding_input_channel_count,       self.enc2.output_channel_count) if self.use_global_track else bi_noop
        self.gte2 = GlobalTrackLayer(2 * self.enc2.output_channel_count, self.enc3.output_channel_count) if self.use_global_track else bi_noop
        self.gte3 = GlobalTrackLayer(2 * self.enc3.output_channel_count, self.enc4.output_channel_count) if self.use_global_track else bi_noop
        self.gte4 = GlobalTrackLayer(2 * self.enc4.output_channel_count, self.enc5.output_channel_count) if self.use_global_track else bi_noop
        self.gte5 = GlobalTrackLayer(2 * self.enc5.output_channel_count, self.enc6.output_channel_count) if self.use_global_track else bi_noop
        self.gte6 = GlobalTrackLayer(2 * self.enc6.output_channel_count, self.enc7.output_channel_count) if self.use_global_track else bi_noop
        self.gte7 = GlobalTrackLayer(2 * self.enc7.output_channel_count, self.enc8.output_channel_count) if self.use_global_track else bi_noop
        self.gte8 = GlobalTrackLayer(2 * self.enc8.output_channel_count, self.dec8.output_channel_count) if self.use_global_track else bi_noop

        self.gtd8 = GlobalTrackLayer(2 * self.dec8.output_channel_count, self.dec7.output_channel_count) if self.use_global_track else bi_noop
        self.gtd7 = GlobalTrackLayer(2 * self.dec7.output_channel_count, self.dec6.output_channel_count) if self.use_global_track else bi_noop
        self.gtd6 = GlobalTrackLayer(2 * self.dec6.output_channel_count, self.dec5.output_channel_count) if self.use_global_track else bi_noop
        self.gtd5 = GlobalTrackLayer(2 * self.dec5.output_channel_count, self.dec4.output_channel_count) if self.use_global_track else bi_noop
        self.gtd4 = GlobalTrackLayer(2 * self.dec4.output_channel_count, self.dec3.output_channel_count) if self.use_global_track else bi_noop
        self.gtd3 = GlobalTrackLayer(2 * self.dec3.output_channel_count, self.dec2.output_channel_count) if self.use_global_track else bi_noop
        self.gtd2 = GlobalTrackLayer(2 * self.dec2.output_channel_count, self.dec1.output_channel_count) if self.use_global_track else bi_noop
        self.gtd1 = GlobalTrackLayer(2 * self.dec1.output_channel_count, self.output_channel_count)      if self.use_global_track else bi_noop

        self.final_activation = torch.nn.Sigmoid()

    def forward(self, input):
        if self.coord is not None:
            input = self.coord(input)

        input_mean = torch.mean(input, dim=(2,3), keepdim=False) if self.use_global_track else None

        # Encoding
        down1, _          = self.enc1(input,      None)
        global_track      = self.gte1(input_mean, None)
        down2, down2_mean = self.enc2(down1,      global_track)
        global_track      = self.gte2(down2_mean, global_track)
        down3, down3_mean = self.enc3(down2,      global_track)
        global_track      = self.gte3(down3_mean, global_track)
        down4, down4_mean = self.enc4(down3,      global_track)
        global_track      = self.gte4(down4_mean, global_track)
        down5, down5_mean = self.enc5(down4,      global_track)
        global_track      = self.gte5(down5_mean, global_track)
        down6, down6_mean = self.enc6(down5,      global_track)
        global_track      = self.gte6(down6_mean, global_track)
        down7, down7_mean = self.enc7(down6,      global_track)
        global_track      = self.gte7(down7_mean, global_track)
        down8, down8_mean = self.enc8(down7,      global_track)
        global_track      = self.gte8(down8_mean, global_track)

        # Decoding
        up8, up8_mean = self.dec8(down8, None, global_track)
        global_track  = self.gtd8(up8_mean,    global_track)
        up7, up7_mean = self.dec7(up8, down7,  global_track)
        global_track  = self.gtd7(up7_mean,    global_track)
        up6, up6_mean = self.dec6(up7, down6,  global_track)
        global_track  = self.gtd6(up6_mean,    global_track)
        up5, up5_mean = self.dec5(up6, down5,  global_track)
        global_track  = self.gtd5(up5_mean,    global_track)
        up4, up4_mean = self.dec4(up5, down4,  global_track)
        global_track  = self.gtd4(up4_mean,    global_track)
        up3, up3_mean = self.dec3(up4, down3,  global_track)
        global_track  = self.gtd3(up3_mean,    global_track)
        up2, up2_mean = self.dec2(up3, down2,  global_track)
        global_track  = self.gtd2(up2_mean,    global_track)
        up1, up1_mean = self.dec1(up2, down1,  global_track)
        global_track  = self.gtd1(up1_mean,    global_track)

        return self.final_activation(up1)

def gamma_decode(images):
    return torch.pow(images, 2.2)

def gamma_encode(images):
    return torch.pow(images, 1.0/2.2)

writer = SummaryWriter("./logs")

generator = Generator(3).cuda() # For 12 we need loss function that weights the individual components individually instead of pushing everything into one L1
print(generator)

img1 = plt.imread("./out1.png")
img2 = plt.imread("./out2.png")
img3 = plt.imread("./out3.png")

def read_data(paths):
    inputs  = None
    targets = None
    for path in paths:
        img  = gamma_decode(torch.Tensor(plt.imread(path)).permute(2, 0, 1))
        imgs = torch.cat(img.unsqueeze(0).split(256, dim=-1), 0)
        input  = imgs[-5].unsqueeze(0)
        target = torch.cat(imgs[10:11].split(1, dim=0), dim=1)
        
        if inputs is None:
            inputs = input
        else:
            inputs = torch.cat((inputs, input), dim=0)

        if targets is None:
            targets = target
        else:
            targets = torch.cat((targets, target), dim=0)

    return inputs, targets

inputs, targets = read_data(["./in1.png", "./in2.png", "./in3.png"])

input = gamma_decode(torch.tensor([img1, img2]).permute(0, 3, 1, 2).cuda())
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
for epoch in range(1000):

    batch_inputs  = inputs[:2].cuda()
    batch_targets = targets[:2].cuda()

    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    outputs = generator(batch_inputs)
    loss = criterion(outputs, batch_targets)
    loss.backward()
    optimizer.step()    # Does the update

    writer.add_scalar("loss", loss.item(), epoch)

    print("Epoch {:d}, loss: {:f}".format(epoch + 1, loss.item()))

writer.add_graph(generator, input)
writer.close()

fig=plt.figure(figsize=(8, 8))
for i in range(inputs.shape[0]):
    input = inputs[i].unsqueeze(0).cuda()
    #target = targets[i]

    output = generator(input)
             
    images = torch.cat(output.split(3, dim=1), dim=0).clone().cpu().detach().permute(0, 2, 3, 1)

    fig.add_subplot(inputs.shape[0], 5, i * 5 + 1)
    plt.imshow(gamma_encode(inputs[i].clone().cpu().detach().permute(1, 2, 0)))

    fig.add_subplot(inputs.shape[0], 5, i * 5 + 2)
    plt.imshow(gamma_encode(images[0]))

    # fig.add_subplot(inputs.shape[0], 5, i * 5 + 3)
    # plt.imshow(gamma_encode(images[1]))

    # fig.add_subplot(inputs.shape[0], 5, i * 5 + 4)
    # plt.imshow(gamma_encode(images[2]))

    # fig.add_subplot(inputs.shape[0], 5, i * 5 + 5)
    # plt.imshow(gamma_encode(images[3]))
plt.show()
