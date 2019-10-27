import tensorflow.compat.v1 as tf
import numpy as np

#Convolution implementation
def conv(batch_input, out_channels, stride, filterSize=4, initScale = 0.02, useXavier=False, paddingSize = 1, useBias=False):
    with tf.variable_scope("conv"):
        in_height, in_width, in_channels = [batch_input.get_shape()[1], batch_input.get_shape()[2], int(batch_input.get_shape()[-1])]
        filter = tf.get_variable("filter", [filterSize, filterSize, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, np.sqrt(2.0/(int(in_channels) + int(out_channels))) * initScale) if useXavier else tf.random_normal_initializer(0, initScale))

        padded_input = tf.pad(batch_input, [[0, 0], [paddingSize, paddingSize], [paddingSize, paddingSize], [0, 0]], mode="CONSTANT")#SYMMETRIC
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")

        if useBias:
            offset = tf.get_variable("offset", [1, 1, 1, out_channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            conv = conv + offset
        return conv


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

#Deconvolution used in the method
def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        in_height, in_width, in_channels = [int(batch_input.get_shape()[1]), int(batch_input.get_shape()[2]), int(batch_input.get_shape()[3])]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        filter1 = tf.get_variable("filter1", [4, 4, out_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))

        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        resized_images = tf.image.resize_images(batch_input, [in_height * 2, in_width * 2], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)#BILINEAR
        conv = tf.nn.conv2d(resized_images, filter, [1, 1, 1, 1], padding="SAME")
        conv = tf.nn.conv2d(conv, filter1, [1, 1, 1, 1], padding="SAME")

        #conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv
        
#Theoritically correct convolution
def deconv_bis(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        in_height, in_width, in_channels = [int(batch_input.get_shape()[1]), int(batch_input.get_shape()[2]), int(batch_input.get_shape()[3])]
        filter = tf.get_variable("filter", [3, 3, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        filter1 = tf.get_variable("filter1", [3, 3, out_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))

        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        resized_images = tf.image.resize_images(batch_input, [in_height * 2, in_width * 2], method = tf.image.ResizeMethod.BILINEAR, align_corners=True)#NEAREST_NEIGHBOR BILINEAR
        paddingSize = 1
        padded = tf.pad(resized_images, [[0, 0], [paddingSize, paddingSize], [paddingSize, paddingSize], [0, 0]], mode="REFLECT")#CONSTANT
        conv = tf.nn.conv2d(padded, filter, [1, 1, 1, 1], padding="VALID")

        padded = tf.pad(conv, [[0, 0], [paddingSize, paddingSize], [paddingSize, paddingSize], [0, 0]], mode="SYMMETRIC")#CONSTANT
        conv = tf.nn.conv2d(padded, filter1, [1, 1, 1, 1], padding="VALID")

        #conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv

#input is of shape [batch, X]. Returns the outputs of the layer. 
def fullyConnected(input, outputDim, useBias, layerName = "fully_connected", initMultiplyer = 1.0):
    with tf.variable_scope(layerName):
        batchSize = tf.shape(input)[0];
        inputChannels = int(input.get_shape()[-1])
        weights = tf.get_variable("weight", [inputChannels, outputDim ], dtype=tf.float32, initializer=tf.random_normal_initializer(0, initMultiplyer * tf.sqrt(1.0/float(inputChannels)))) #TODO Is this init a good idea ?
        weightsTiled = tf.tile(tf.expand_dims(weights, axis = 0), [batchSize, 1,1])
        squeezedInput = input

        if (len(input.get_shape()) > 3) :
            squeezedInput = tf.squeeze(squeezedInput, [1])
            squeezedInput = tf.squeeze(squeezedInput, [1])

        outputs = tf.matmul(tf.expand_dims(squeezedInput, axis = 1), weightsTiled)
        outputs = tf.squeeze(outputs, [1])
        if(useBias):
            bias = tf.get_variable("bias", [outputDim], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.002))
            outputs = outputs + tf.expand_dims(bias, axis = 0)

        return outputs

#Takes a globalGenerator output as input and transforms it so it can be added to the main U-NET track
def GlobalToGenerator(inputs, channels):
    with tf.variable_scope("GlobalToGenerator1"):
        fc1 = fullyConnected(inputs, channels, False, "fullyConnected_global_to_unet" ,0.01) #Why so low ?
    return tf.expand_dims(tf.expand_dims(fc1, axis = 1), axis=1)

#Pooling implementation (max or mean depending on what is requested by the user)
def pooling(pooling_type, inputs, dynamic_batch_size):
    outputs = inputs#tf.reshape(inputs, [dynamic_batch_size, -1, int(inputs.get_shape()[1]), int(inputs.get_shape()[2]), int(inputs.get_shape()[3])])
    # outputs should be [batch, nbInputs, W, H,C]
    print(pooling_type)
    if pooling_type == "max":
        outputs = tf.reduce_max(outputs, axis=1)
    elif pooling_type == "mean":
        outputs = tf.reduce_mean(outputs, axis=1)
    #outputs should be [batch, W, H, C]
    tf.Print(outputs, [tf.shape(outputs)], "outputs shape after pooling: ")
    return outputs

#The instance normalization implementation
def instancenorm(input):
    with tf.variable_scope("instancenorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [1, 1, 1, channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [1, 1, 1, channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        #[batchsize ,1,1, channelNb]
        variance_epsilon = 1e-5
        #Batch normalization function does the mean substraction then divide by the standard deviation (to normalize it). It finally multiply by scale and adds offset.
        #normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        #For instanceNorm we do it ourselves :
        normalized = (((input - mean) / tf.sqrt(variance + variance_epsilon)) * scale) + offset
        return normalized, mean, variance
