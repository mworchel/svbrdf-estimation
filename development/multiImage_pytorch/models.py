import torch
import torch.nn as nn
import utils

class LayerBootstrapping:
    def __init__(self, use_convolution_bias=False, use_linear_bias=False, initialize_weights=True, convolution_init_scale=0.02, linear_init_scale=0.01):
        self.use_convolution_bias   = use_convolution_bias
        self.use_linear_bias        = use_linear_bias
        self.initialize_weights     = initialize_weights
        self.convolution_init_scale = convolution_init_scale
        self.linear_init_scale      = linear_init_scale

    def initialize_module(self, m):
        if self.initialize_weights:
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, 0.0, self.linear_init_scale * torch.sqrt(torch.tensor(1.0 / float(m.in_features))))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif type(m) == nn.Conv2d:
                torch.nn.init.normal_(m.weight, 0.0, self.convolution_init_scale)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        return m

class EncodingLayer(nn.Module):
    def __init__(self, bootstrapping, input_channel_count, output_channel_count, use_instance_norm, use_activation=True):
        super(EncodingLayer, self).__init__()
        
        self.input_channel_count  = input_channel_count
        self.output_channel_count = output_channel_count
        self.use_instance_norm    = use_instance_norm
        self.use_activation       = use_activation

        self.conv            = nn.Conv2d(input_channel_count, output_channel_count, (4, 4), stride=2, padding=(1,1), bias=bootstrapping.use_convolution_bias)
        self.norm            = torch.nn.InstanceNorm2d(output_channel_count, 1e-5, affine=True) if use_instance_norm else None
        self.leaky_relu      = nn.LeakyReLU(0.2) if use_activation else None
        self.fully_connected = nn.Linear(self.output_channel_count, self.output_channel_count, bias=bootstrapping.use_linear_bias)

        self.apply(bootstrapping.initialize_module)

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
    def __init__(self, bootstrapping, input_channel_count, output_channel_count, use_instance_norm, use_dropout, use_activation=True):
        super(DecodingLayer, self).__init__()
        
        self.input_channel_count  = input_channel_count
        self.output_channel_count = output_channel_count
        self.use_instance_norm    = use_instance_norm
        self.use_dropout          = use_dropout
        self.use_activation       = use_activation

        self.deconv     = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2.0),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(input_channel_count,  output_channel_count, (4, 4), bias=bootstrapping.use_convolution_bias),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(output_channel_count, output_channel_count, (4, 4), bias=bootstrapping.use_convolution_bias),
        )
        
        self.norm            = torch.nn.InstanceNorm2d(output_channel_count, 1e-5, affine=True) if use_instance_norm else None
        self.dropout         = nn.Dropout(0.5) if use_dropout else None
        self.leaky_relu      = nn.LeakyReLU(0.2) if use_activation else None
        self.fully_connected = nn.Linear(self.output_channel_count, self.output_channel_count, bias=bootstrapping.use_linear_bias)

        self.apply(bootstrapping.initialize_module)

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
    def __init__(self, bootstrapping, input_channel_count, output_channel_count):
        super(GlobalTrackLayer, self).__init__()

        self.input_channel_count = input_channel_count
        self.output_channel_count = output_channel_count

        self.fully_connected = nn.Linear(input_channel_count, output_channel_count, bias=bootstrapping.use_linear_bias)
        self.selu            = nn.SELU()

        self.apply(bootstrapping.initialize_module)

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

        self.coord = CoordLayer() if self.use_coords else None   

        # Use a bootstrapper for sharing setup and initialization of convolutional and linear layers in the encoder/decoder
        # The reference implementation uses the following variables and values
        # useBias        = False (for conv and linear) [default:True ]
        # init           = True                        [default:False]
        # initScale      = 0.02  (for conv) 
        # initMultiplier = 0.01  (for linear)
        # encdec_bootstrap = LayerBootstrapping(use_convolution_bias=False, use_linear_bias=False, initialize_weights=True, convolution_init_scale=0.02, linear_init_scale=0.01)
        # FIXME: I personally have the feeling that not performing explicit initialization is better for convergence
        encdec_bootstrap = LayerBootstrapping(use_convolution_bias=False, use_linear_bias=False, initialize_weights=False, convolution_init_scale=0.02, linear_init_scale=0.01)

        encoding_input_channel_count = self.input_channel_count + 2 if self.use_coords else self.input_channel_count
        self.enc1 = EncodingLayer(encdec_bootstrap, encoding_input_channel_count,   self.number_of_filters    , False, False) # encoder_1: [batch, 256, 256, 3      ] => [batch, 128, 128, ngf    ]
        self.enc2 = EncodingLayer(encdec_bootstrap, self.enc1.output_channel_count, self.number_of_filters * 2,  True)        # encoder_2: [batch, 128, 128, ngf    ] => [batch,  64,  64, ngf * 2]
        self.enc3 = EncodingLayer(encdec_bootstrap, self.enc2.output_channel_count, self.number_of_filters * 4,  True)        # encoder_3: [batch,  64,  64, ngf * 2] => [batch,  32,  32, ngf * 4]
        self.enc4 = EncodingLayer(encdec_bootstrap, self.enc3.output_channel_count, self.number_of_filters * 8,  True)        # encoder_4: [batch,  32,  32, ngf * 4] => [batch,  16,  16, ngf * 8]
        self.enc5 = EncodingLayer(encdec_bootstrap, self.enc4.output_channel_count, self.number_of_filters * 8,  True)        # encoder_5: [batch,  16,  16, ngf * 8] => [batch,   8,   8, ngf * 8]
        self.enc6 = EncodingLayer(encdec_bootstrap, self.enc5.output_channel_count, self.number_of_filters * 8,  True)        # encoder_6: [batch,   8,   8, ngf * 8] => [batch,   4,   4, ngf * 8]
        self.enc7 = EncodingLayer(encdec_bootstrap, self.enc6.output_channel_count, self.number_of_filters * 8,  True)        # encoder_7: [batch,   4,   4, ngf * 8] => [batch,   2,   2, ngf * 8]
        self.enc8 = EncodingLayer(encdec_bootstrap, self.enc6.output_channel_count, self.number_of_filters * 8, False)        # encoder_8: [batch,   2,   2, ngf * 8] => [batch,   1,   1, ngf * 8]                  

        self.dec8 = DecodingLayer(encdec_bootstrap, self.number_of_filters * 8,         self.number_of_filters * 8,  True,  True) # decoder_8: [batch,  1,  1,       ngf * 8] => [batch,   2,   2, ngf * 8]
        self.dec7 = DecodingLayer(encdec_bootstrap, 2 * self.dec8.output_channel_count, self.number_of_filters * 8,  True,  True) # decoder_7: [batch,  2,  2,   2 * ngf * 8 ] => [batch,   4,   4, ngf * 8]
        self.dec6 = DecodingLayer(encdec_bootstrap, 2 * self.dec7.output_channel_count, self.number_of_filters * 8,  True,  True) # decoder_6: [batch,  4,  4,   2 * ngf * 8 ] => [batch,   8,   8, ngf * 8] 
        self.dec5 = DecodingLayer(encdec_bootstrap, 2 * self.dec6.output_channel_count, self.number_of_filters * 8,  True, False) # decoder_5: [batch,  8,  8,   2 * ngf * 8 ] => [batch,  16,  16, ngf * 8]
        self.dec4 = DecodingLayer(encdec_bootstrap, 2 * self.dec5.output_channel_count, self.number_of_filters * 4,  True, False) # decoder_4: [batch, 16, 16,   2 * ngf * 8 ] => [batch,  32,  32, ngf * 4]
        self.dec3 = DecodingLayer(encdec_bootstrap, 2 * self.dec4.output_channel_count, self.number_of_filters * 2,  True, False) # decoder_3: [batch, 32, 32,   2 * ngf * 4 ] => [batch,  64,  64, ngf * 2]
        self.dec2 = DecodingLayer(encdec_bootstrap, 2 * self.dec3.output_channel_count, self.number_of_filters    ,  True, False) # decoder_2: [batch, 64, 64,   2 * ngf * 2 ] => [batch, 128, 128, ngf    ]
        self.dec1 = DecodingLayer(encdec_bootstrap, 2 * self.dec2.output_channel_count, self.output_channel_count , False, False) # decoder_1: [batch, 128, 128, 2 * ngf     ] => [batch, 256, 256, 64     ]                   

        # Use a bootstrapper for sharing setup and initialization of convolutional and linear layers in the global track
        # The reference implementation uses the following variables and values (convolutional layers are not created)
        # useBias        = True  (for linear) [default:True ]
        # init           = True               [default:False]
        # initMultiplier = 1.0   (for linear)
        # gt_boostrap = ConvLinBootstrap(use_linear_bias=True, initialize_weights=True, linear_init_scale=1.0)
        # FIXME: I personally have the feeling that not performing explicit initialization is better for convergence
        gt_boostrap = LayerBootstrapping(use_linear_bias=True, initialize_weights=False, linear_init_scale=1.0)

        def bi_noop(x, y):
            return None

        self.gte1 = GlobalTrackLayer(gt_boostrap, encoding_input_channel_count,       self.enc2.output_channel_count) if self.use_global_track else bi_noop
        self.gte2 = GlobalTrackLayer(gt_boostrap, 2 * self.enc2.output_channel_count, self.enc3.output_channel_count) if self.use_global_track else bi_noop
        self.gte3 = GlobalTrackLayer(gt_boostrap, 2 * self.enc3.output_channel_count, self.enc4.output_channel_count) if self.use_global_track else bi_noop
        self.gte4 = GlobalTrackLayer(gt_boostrap, 2 * self.enc4.output_channel_count, self.enc5.output_channel_count) if self.use_global_track else bi_noop
        self.gte5 = GlobalTrackLayer(gt_boostrap, 2 * self.enc5.output_channel_count, self.enc6.output_channel_count) if self.use_global_track else bi_noop
        self.gte6 = GlobalTrackLayer(gt_boostrap, 2 * self.enc6.output_channel_count, self.enc7.output_channel_count) if self.use_global_track else bi_noop
        self.gte7 = GlobalTrackLayer(gt_boostrap, 2 * self.enc7.output_channel_count, self.enc8.output_channel_count) if self.use_global_track else bi_noop
        self.gte8 = GlobalTrackLayer(gt_boostrap, 2 * self.enc8.output_channel_count, self.dec8.output_channel_count) if self.use_global_track else bi_noop

        self.gtd8 = GlobalTrackLayer(gt_boostrap, 2 * self.dec8.output_channel_count, self.dec7.output_channel_count) if self.use_global_track else bi_noop
        self.gtd7 = GlobalTrackLayer(gt_boostrap, 2 * self.dec7.output_channel_count, self.dec6.output_channel_count) if self.use_global_track else bi_noop
        self.gtd6 = GlobalTrackLayer(gt_boostrap, 2 * self.dec6.output_channel_count, self.dec5.output_channel_count) if self.use_global_track else bi_noop
        self.gtd5 = GlobalTrackLayer(gt_boostrap, 2 * self.dec5.output_channel_count, self.dec4.output_channel_count) if self.use_global_track else bi_noop
        self.gtd4 = GlobalTrackLayer(gt_boostrap, 2 * self.dec4.output_channel_count, self.dec3.output_channel_count) if self.use_global_track else bi_noop
        self.gtd3 = GlobalTrackLayer(gt_boostrap, 2 * self.dec3.output_channel_count, self.dec2.output_channel_count) if self.use_global_track else bi_noop
        self.gtd2 = GlobalTrackLayer(gt_boostrap, 2 * self.dec2.output_channel_count, self.dec1.output_channel_count) if self.use_global_track else bi_noop
        self.gtd1 = GlobalTrackLayer(gt_boostrap, 2 * self.dec1.output_channel_count, self.output_channel_count)      if self.use_global_track else bi_noop

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

        return up1, global_track

class SingleViewModel(nn.Module):
    def __init__(self):
        super(SingleViewModel, self).__init__()

        self.generator  = Generator(9)
        self.activation = nn.Tanh()

    def forward(self, input):
        if len(input.shape) == 5:
            # If we get multiple input images, we just ignore all but one
            input = input[:,0,:,:,:]

        svbrdf, _ = self.generator(input)
        svbrdf    = self.activation(svbrdf)

        # 9 channel SVBRDF to 12 channels
        svbrdf    = utils.decode_svbrdf(svbrdf) 

        # Map ranges from [-1, 1] to [0, 1], except for the normals
        normals, diffuse, roughness, specular = utils.unpack_svbrdf(svbrdf)
        diffuse   = utils.encode_as_unit_interval(diffuse)
        roughness = utils.encode_as_unit_interval(roughness)
        specular  = utils.encode_as_unit_interval(specular)

        return utils.pack_svbrdf(normals, diffuse, roughness, specular)

class MultiViewModel(nn.Module):
    def __init__(self):
        super(MultiViewModel, self).__init__()

        # Create the generator
        self.generator = Generator(64)

        # TODO: Fusion (Pooling and conv features)

    def forward(self, input):
        # Split the input of shape (B, N, C, H, W) into a list over the input images [(B, 1, C, H, W)_1, ..., (B, 1, C, H, W)_N]
        input_images = torch.split(input, 1, dim=1) 

        # Invoke the generator for all the input images
        encoder_decoder_outputs = []
        global_track_outputs = []
        for input_image in input_images:
            encoder_decoder_output, global_track_output = self.generator(input_image.squeeze(1))
            encoder_decoder_outputs.append(encoder_decoder_output.unsqueeze(1))
            global_track_outputs.append(global_track_output.unsqueeze(1))
            batch_outputs = batch_output if batch_outputs.unsqueeze(1) is None else torch.cat([batch_outputs, batch_output], dim=1)

        # Merge the outputs back into a tensors of shape (B, N, C, H, W)
        encoder_decoder_outputs = torch.cat(encoder_decoder_outputs, dim=1)
        global_track_outputs    = torch.cat(global_track_outputs, dim=1)

        # Pool over the input image dimension
        pooled_encoder_decoder_outputs = torch.max(encoder_decoder_outputs, dim=1)
        pooled_global_track_outputs    = torch.max(global_track_outputs, dim=1)

        # TODO: Feature extraction and activation...

        # TODO: Deprocess images (SVBRDF decoding)
        