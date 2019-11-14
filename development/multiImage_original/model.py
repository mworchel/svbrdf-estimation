import tensorflow.compat.v1 as tf
import tfHelpers
import helpers
#Define the the model class, this contains our network definition. Quite similar to the first project.
class Model:
    generatorOutputs = None
    output = None
    inputTensor = None
    useSecondary = True
    useCoordConv = False
    ngf = 64 #Number of filter for the generator
    generatorOutputChannels = 64
    reuse_bool=False
    last_convolutions_channels =[64,32,9]
    pooling_type = "max"
    dynamic_batch_size = None
    firstAsGuide = False
    NoMaxPooling = False

    def __init__(self, input, dyn_batch_size, useSecondary=True, ngf=64, generatorOutputChannels=64, reuse_bool=False,  pooling_type="max", last_convolutions_channels=[64,32,9], useCoordConv = False, firstAsGuide = False, NoMaxPooling = False):
        self.inputTensor = input
        self.useSecondary = useSecondary
        self.ngf = ngf
        self.generatorOutputChannels = generatorOutputChannels
        self.reuse_bool = reuse_bool
        self.pooling_type = pooling_type
        self.last_convolutions_channels = last_convolutions_channels
        self.dynamic_batch_size = dyn_batch_size
        self.useCoordConv = useCoordConv
        self.firstAsGuide = firstAsGuide
        self.NoMaxPooling = NoMaxPooling

    #Secondary network block, used in the submission
    def _addSecondaryNetBlock(self, input, inputMean, lastGlobalNetworkValue, currentChannels, nextChannels, layerCount, keep_dims=False):
        if self.useSecondary:
            if inputMean is None:
                inputMean, _ = tf.nn.moments(input, axes=[1, 2], keep_dims=keep_dims)
            summed = input
            if not lastGlobalNetworkValue is None:
                summed = input + tfHelpers.GlobalToGenerator(lastGlobalNetworkValue, currentChannels)
            with tf.variable_scope("globalNetwork_fc_%d" % (layerCount + 1)):
                nextGlobalInput = inputMean
                if not lastGlobalNetworkValue is None:
                    nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(lastGlobalNetworkValue, axis = 1), axis=1), inputMean], axis = -1)
                globalNetwork_fc = tfHelpers.fullyConnected(nextGlobalInput, nextChannels, True, "globalNetworkLayer" + str(layerCount + 1))

            return summed, tf.nn.selu(globalNetwork_fc) #returns the sum of this layer + last globalNet output and a new globalNetValue
        else:
            return input, None

    #Encoder of the generator, used in the submission
    def __create_encoder(self, input):
        layers = []
        #input shape is [batch * nbRenderings, height, width, 3]
        if self.useCoordConv:
            coords = helpers.generateCoords(tf.shape(input))
            input = tf.concat([input, coords], axis = -1)

        _, lastGlobalNet = self._addSecondaryNetBlock(input, None, None, None ,self.ngf * 2, 1)
        with tf.variable_scope("encoder_1"):
            output = tfHelpers.conv(input, self.ngf, stride=2, useXavier=False)
            layers.append(output)
        #Default ngf is 64
        layer_specs = [
            self.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            self.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            self.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            self.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            self.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            self.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ]
        for layerCount, out_channels in enumerate(layer_specs):
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = tfHelpers.lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = tfHelpers.conv(rectified, out_channels, stride=2, useXavier=False)
                #here mean and variance will be [batch, 1, 1, out_channels]
                outputs, mean, variance = tfHelpers.instancenorm(convolved)
                layers_specs_GlobalNet = layerCount + 1
                if layerCount + 1 >= len(layer_specs) - 1:
                    layers_specs_GlobalNet = layerCount
                outputs, lastGlobalNet = self._addSecondaryNetBlock(outputs, mean, lastGlobalNet, out_channels, layer_specs[layers_specs_GlobalNet], len(layers) + 1)

                layers.append(outputs)

        with tf.variable_scope("encoder_8"):
            rectified = tfHelpers.lrelu(layers[-1], 0.2)
             # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = tfHelpers.conv(rectified, self.ngf * 8, stride=2, useXavier=False)
            convolved, lastGlobalNet = self._addSecondaryNetBlock(convolved, None, lastGlobalNet, self.ngf * 8, self.ngf * 8, len(layers) + 1, keep_dims=True)
            layers.append(convolved)
        return layers, lastGlobalNet
    
    #Decoder of the generator, used in the submission
    def __create_decoder(self, encoder_results, lastGlobalNet, output_channels):
        layer_specs = [
            (self.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8]
            (self.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 ] => [batch, 4, 4, ngf * 8]
            (self.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 ] => [batch, 8, 8, ngf * 8] #Dropout was 0.5 until here
            (self.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 ] => [batch, 16, 16, ngf * 8]
            (self.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 ] => [batch, 32, 32, ngf * 4]
            (self.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4] => [batch, 64, 64, ngf * 2]
            (self.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2] => [batch, 128, 128, ngf]
        ]
        decoder_results = []

        num_encoder_layers = len(encoder_results)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = encoder_results[-1]
                else:
                    input = tf.concat([decoder_results[-1], encoder_results[skip_layer]], axis=3)

                rectified = tfHelpers.lrelu(input, 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = tfHelpers.deconv(rectified, out_channels)
                output, mean, variance = tfHelpers.instancenorm(output)
                outputs, lastGlobalNet = self._addSecondaryNetBlock(output, mean, lastGlobalNet, out_channels, out_channels, num_encoder_layers + len(decoder_results))
                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                decoder_results.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, output_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([decoder_results[-1], encoder_results[0]], axis=3)
            rectified = tfHelpers.lrelu(input, 0.2)
            deconved = tfHelpers.deconv(rectified, output_channels)
            #should we normalize it ?
            deconved, lastGlobalNet = self._addSecondaryNetBlock(deconved, None, lastGlobalNet, output_channels, output_channels, num_encoder_layers + len(decoder_results), True)
            #output = tf.tanh(deconved)
            decoder_results.append(deconved)

        return decoder_results[-1], lastGlobalNet


    #This function creates the convolutional neural net taking as input the pooled features from the generator.
    def __createLastConvs(self, input, secondaryNet_input, output_channels, last_channels, reuse_bool = True):
        input, lastGlobalNet = self._addSecondaryNetBlock(input, None, secondaryNet_input, last_channels, output_channels[0], 0, True)
        #if self.useCoordConv:
        #    coords = helpers.generateCoords(tf.shape(input))
        #    input = tf.concat([input, coords], axis = -1)
        layers = [input]
        for layerCount, chanCount in enumerate(output_channels[:-1]):
            with tf.variable_scope("final_conv_" + str(layerCount)):
                convolved = tfHelpers.conv(layers[-1], chanCount, stride=1, filterSize=3, initScale=0.02, useXavier=False, paddingSize = 1)
                lastLayerResult, mean, variance = tfHelpers.instancenorm(convolved)

                lastLayerResult, lastGlobalNet = self._addSecondaryNetBlock(lastLayerResult, mean, lastGlobalNet, chanCount, output_channels[layerCount + 1], len(layers))
                rectified = tfHelpers.lrelu(lastLayerResult, 0.2)
                layers.append(rectified)
        with tf.variable_scope("final_conv_last"):

            convolved = tfHelpers.conv(layers[-1], output_channels[-1], stride=1, filterSize=3, initScale=0.02, useXavier=True, paddingSize = 1,useBias= True)
            #convolved, _ = self._addSecondaryNetBlock(convolved, None, lastGlobalNet, output_channels[-1], output_channels[-1], len(layers), True)

            outputs = tf.tanh(convolved)
            #outputs should be [batch, W, H, C]
            return outputs
            
    #Another implementation of the generator (copied from the one image project) as I wonder if there is not a small bug in the other implementation.
    def create_generator(self, generator_inputs, generator_outputs_channels, reuse_bool = True):
        with tf.variable_scope("generator", reuse=reuse_bool) as scope:
            #Print the shape to check we are inputting a tensor with a reasonable shape
            print("generator_inputs :" + str(generator_inputs.get_shape()))
            print("generator_outputs_channels :" + str(generator_outputs_channels))
            layers = []
            #Input here should be [batch, 256,256,3]
            inputMean, inputVariance = tf.nn.moments(generator_inputs, axes=[1, 2], keep_dims=False)
            globalNetworkInput = inputMean
            globalNetworkOutputs = []
            with tf.variable_scope("globalNetwork_fc_1"):
                globalNetwork_fc_1 = tfHelpers.fullyConnected(globalNetworkInput, self.ngf * 2, True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc_1))

            #encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
            with tf.variable_scope("encoder_1"):
                #Convolution with stride 2 and kernel size 4x4.
                output = tfHelpers.conv(generator_inputs, self.ngf , stride=2)
                layers.append(output)
            #Default ngf is 64
            layer_specs = [
                self.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
                self.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
                self.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                self.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                self.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                self.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
                #self.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
            ]

            for layerCount, out_channels in enumerate(layer_specs):
                with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                    #We use a leaky relu instead of a relu to let a bit more expressivity to the network.
                    rectified = tfHelpers.lrelu(layers[-1], 0.2)
                    # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                    convolved = tfHelpers.conv(rectified, out_channels, stride=2)
                    #here mean and variance will be [batch, 1, 1, out_channels] and we run an instance normalization
                    outputs, mean, variance = tfHelpers.instancenorm(convolved)
                    
                    #Get the last value in the global feature secondary network and transform it to be added to the current Unet layer output.
                    outputs = outputs + tfHelpers.GlobalToGenerator(globalNetworkOutputs[-1], out_channels)
                    with tf.variable_scope("globalNetwork_fc_%d" % (len(globalNetworkOutputs) + 1)):
                        #Prepare the input to the next global feature secondary network step and run it.
                        nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(globalNetworkOutputs[-1], axis = 1), axis=1), mean], axis = -1)
                        globalNetwork_fc = ""
                        if layerCount + 1 < len(layer_specs) - 1:
                            globalNetwork_fc = tfHelpers.fullyConnected(nextGlobalInput, layer_specs[layerCount + 1], True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                        else :
                            globalNetwork_fc = tfHelpers.fullyConnected(nextGlobalInput, layer_specs[layerCount], True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                        #We use selu as we are in a fully connected network and it has auto normalization properties.
                        globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc))
                    layers.append(outputs)

            with tf.variable_scope("encoder_8"):
                #The last encoder is mostly similar to previous layers except that we don't normalize the output.
                rectified = tfHelpers.lrelu(layers[-1], 0.2)
                 # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolvedNoGlobal = tfHelpers.conv(rectified, self.ngf * 8, stride=2)
                convolved = convolved  + tfHelpers.GlobalToGenerator(globalNetworkOutputs[-1], self.ngf * 8)

                with tf.variable_scope("globalNetwork_fc_%d" % (len(globalNetworkOutputs) + 1)):
                    mean, variance = tf.nn.moments(convolvedNoGlobal, axes=[1, 2], keep_dims=True)
                    nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(globalNetworkOutputs[-1], axis = 1), axis=1), mean], axis = -1)
                    globalNetwork_fc = tfHelpers.fullyConnected(nextGlobalInput, self.ngf * 8, True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                    globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc))

                layers.append(convolved)
            #default nfg = 64
            layer_specs = [
                (self.ngf * 8 , 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (self.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (self.ngf * 8 , 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
                (self.ngf * 8 , 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
                (self.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
                (self.ngf * 2 , 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
                (self.ngf , 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
            ]
            #Start the decoder here
            num_encoder_layers = len(layers)
            for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                #Evaluate which layer from the encoder has to be contatenated for the skip connection
                with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                    if decoder_layer == 0:
                        # first decoder layer doesn't have skip connections
                        # since it is directly connected to the skip_layer
                        input = layers[-1]
                    else:
                        input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                    
                    #Leaky relu some more (same reason as in the encoder)
                    rectified = tfHelpers.lrelu(input, 0.2)
                    # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                    
                    #The deconvolution has stride 1 and shape 4x4. Theorically, it should be shape 3x3 to avoid any effects on the image borders, but it doesn't seem to have such a strong effect.
                    output = tfHelpers.deconv(rectified, out_channels)
                    
                    #Instance norm and global feature secondary network similar to the decoder.
                    output, mean, variance = tfHelpers.instancenorm(output)
                    output = output + tfHelpers.GlobalToGenerator(globalNetworkOutputs[-1], out_channels)
                    with tf.variable_scope("globalNetwork_fc_%d" % (len(globalNetworkOutputs) + 1)):
                        nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(globalNetworkOutputs[-1], axis = 1), axis=1), mean], axis = -1)
                        globalNetwork_fc = tfHelpers.fullyConnected(nextGlobalInput, out_channels, True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                        globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc))
                    if dropout > 0.0:
                        #We use dropout as described in the pix2pix paper.
                        output = tf.nn.dropout(output, keep_prob=1 - dropout)

                    layers.append(output)

            # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
            with tf.variable_scope("decoder_1"):
                input = tf.concat([layers[-1], layers[0]], axis=3)
                rectified = tfHelpers.lrelu(input, 0.2)
                output = tfHelpers.deconv(rectified, generator_outputs_channels)
                lastGlobalNet = tfHelpers.GlobalToGenerator(globalNetworkOutputs[-1], generator_outputs_channels)
                output = output + lastGlobalNet
                #output = tf.tanh(output)
                layers.append(output)

            return layers[-1], lastGlobalNet
    
    #call the proper functions to create the generator.
    def __create_generator(self, input, output_channels, reuse_bool = True):
        with tf.variable_scope("generator", reuse=reuse_bool) as scope:
            #generator_output, secondary_output = self.create_generator(input, output_channels, reuse_bool)
            encoder_results, lastGlobalNet = self.__create_encoder(input)
            decoder_results, lastGlobalNet = self.__create_decoder(encoder_results, lastGlobalNet, output_channels)
            #output = tf.tanh(decoder_results)
            generator_output = decoder_results
        return generator_output, lastGlobalNet

    #create the full model
    def create_model(self):
        with tf.variable_scope("trainableModel", reuse=self.reuse_bool) as scope:
            #get all the generator outputs
            generator_output, secondary_output = self.__create_generator(self.inputTensor, self.generatorOutputChannels, self.reuse_bool)
            pooledGeneratorOutput = generator_output

            pooledSecondaryOutput = secondary_output
            #If no max pooling all images are treated separately. else, process all images and pull them.
            if not self.NoMaxPooling:
                #Separate again the dimension of the batch and the dimension of the number of images.
                tmpOutputs = tf.reshape(pooledGeneratorOutput, [self.dynamic_batch_size, -1, tf.shape(generator_output)[1], tf.shape(generator_output)[2], int(generator_output.get_shape()[3])])
                tmpSecondary = tf.reshape(pooledSecondaryOutput, [self.dynamic_batch_size, -1, int(secondary_output.get_shape()[1])])
                #if the first image should be used as guide concat the pooled value of the last items and the non pooled of the first item
                #If we use the first image as a guide for the full acquisition, process it separately
                if self.firstAsGuide:
                    self.generatorOutputChannels = self.generatorOutputChannels * 2
                    pooledGeneratorOutput = tfHelpers.pooling(self.pooling_type, tmpOutputs, self.dynamic_batch_size)
                    pooledGeneratorOutput = tf.concat([tmpOutputs[:,0], pooledGeneratorOutput], axis = -1) #should now have twice the nb of channels

                    pooledSecondaryOutput = tfHelpers.pooling(self.pooling_type, tmpSecondary, self.dynamic_batch_size)
                    pooledSecondaryOutput = tf.concat([tmpSecondary[:,0], pooledSecondaryOutput], axis = -1) #should now have twice the nb of channels
                #If we pool all the results (default mode), we simply use the user defined pooling operation to collapse the dimension with the different image number.
                else:
                    pooledGeneratorOutput = tfHelpers.pooling(self.pooling_type, tmpOutputs, self.dynamic_batch_size)
                    pooledSecondaryOutput = tfHelpers.pooling(self.pooling_type, tmpSecondary, self.dynamic_batch_size)
            
            #Create the final convolutions to process the pooled features and output the maps
            partialOutput = self.__createLastConvs(pooledGeneratorOutput, pooledSecondaryOutput, self.last_convolutions_channels, self.generatorOutputChannels)
            
            #Process the outputs to have 3 channels for all parameter maps.
            self.output = helpers.deprocess_outputs(partialOutput)
