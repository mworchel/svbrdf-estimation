import os
import tensorflow.compat.v1 as tf
import numpy as np
import math
import acquisitionScene
#import renderer

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

#Log a tensor and normalize it.
def logTensor(tensor):
    return  (tf.log(tf.add(tensor,0.01)) - tf.log(0.01)) / (tf.log(1.01)-tf.log(0.01))

#Generate a random direction on the upper hemisphere with gaps on the top and bottom of Hemisphere. Equation is described in the Global Illumination Compendium (19a)
def tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.05):
    r1 = tf.random_uniform([batchSize, nbRenderings, 1], 0.0 + lowEps, 1.0 - highEps, dtype=tf.float32)
    r2 =  tf.random_uniform([batchSize, nbRenderings, 1], 0.0, 1.0, dtype=tf.float32)
    r = tf.sqrt(r1)
    phi = 2 * math.pi * r2
    #min alpha = atan(sqrt(1-r^2)/r)
    x = r * tf.cos(phi)
    y = r * tf.sin(phi)
    z = tf.sqrt(1.0 - tf.square(r))
    finalVec = tf.concat([x, y, z], axis=-1) #Dimension here should be [batchSize,nbRenderings, 3]
    return finalVec

#Remove the gamma of a vector
def removeGamma(tensor):
    return tf.pow(tensor, 2.2)

#Add gamma to a vector
def addGamma(tensor):
    return tf.pow(tensor, 0.4545)

def target_reshape(targetBatch):
    #Here the target batch is [?(Batchsize), 4, 256, 256, 3] and we want to go to [?(Batchsize), 256,256,12]
    return tf.concat(tf.unstack(targetBatch, axis = 1), axis = -1)

def target_deshape(target, nbTargets):
    #ŧarget have shape [batchsize, 256,256,12] and we want [batchSize,4, 256,256,3]
    target_list = tf.split(target, nbTargets, axis=-1)#4 * [batch, 256,256,3]
    return tf.stack(target_list, axis = 1) #[batch, 4,256,256,3]

#Generate a distance to compute for the specular renderings (as position is important for this kind of renderings)
def tf_generate_distance(batchSize, nbRenderings):
    gaussian = tf.random_normal([batchSize, nbRenderings, 1], 0.5, 0.75, dtype=tf.float32) # parameters chosen empirically to have a nice distance from a -1;1 surface.
    return (tf.exp(gaussian))

#Normalize a tensor
def NormalizeIntensity(tensor):
    maxValue = tf.reduce_max(tensor)
    return tensor / maxValue

# Normalizes a tensor troughout the Channels dimension (BatchSize, Width, Height, Channels)
# Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def tf_Normalize(tensor):
    Length = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis = -1, keep_dims=True))
    return tf.div(tensor, Length)

# Computes the dot product between 2 tensors (BatchSize, Width, Height, Channels)
# Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def tf_DotProduct(tensorA, tensorB):
    return tf.reduce_sum(tf.multiply(tensorA, tensorB), axis = -1, keep_dims=True)

#Very small lamp attenuation
def tf_lampAttenuation(distance):
    DISTANCE_ATTENUATION_MULT = 0.001
    return 1.0 / (1.0 + DISTANCE_ATTENUATION_MULT*tf.square(distance))

#Physically based lamp attenuation
def tf_lampAttenuation_pbr(distance):
    return 1.0 / tf.square(distance)

#Clip values between min an max
def squeezeValues(tensor, min, max):
    return tf.clip_by_value(tensor, min, max)

#Make sure the roughness is between 0.1 and 1.0 to avoid having pure mirror (with nothing to reflect it will be black)
def adaptRougness(mixedMaterial):
    #Material has shape [Batch, 4, 256,256,3]
    multiplier = [1.0, 1.0, 0.9, 1.0]
    addition = [0.0, 0.0, 0.1, 0.0]
    multiplier = tf.reshape(multiplier, [1,4,1,1,1])
    addition = tf.reshape(addition, [1,4,1,1,1])
    return (mixedMaterial * multiplier) + addition

#Varies a bit the power of albedos to augment the variety
def adaptAlbedos(mixedMaterial, batchSize):
    randomAlbedoMultipliers = tf.random_uniform([batchSize, 2], 0.7, 1.0)
    multiplier = tf.concat([tf.ones([batchSize, 1]), tf.expand_dims(randomAlbedoMultipliers[:,0], axis=-1), tf.ones([batchSize, 1]), tf.expand_dims(randomAlbedoMultipliers[:,1], axis=-1)], axis = -1)
    multiplier = tf.reshape(multiplier, [batchSize,4,1,1,1])
    return (mixedMaterial * multiplier)

#Flatten normals in a target tensor to augment the variety
def adaptNormals(mixedMaterial, batchSize):
    shouldRemoveNormal = tf.random_uniform([batchSize], 0.0, 5.0)
    boolValue = tf.floor(tf.minimum(shouldRemoveNormal, 1.0)) #0 if shouldRemoveNormal>=0 && <= 1 ; 1 if > 1. So it has (100/5) % of chance to happen
    boolValue = tf.reshape(boolValue, [batchSize, 1, 1, 1])

    normals = preprocess(mixedMaterial[:,0]) #go to -1;1
    normalsXY = normals[:,:,:,:-1] * boolValue #put to 0 the XY if needed
    normals = tf.concat([normalsXY, tf.expand_dims(normals[:,:,:,-1], axis=-1)], axis = -1)
    normals = tf_Normalize(normals) #renormalize, putting the Z to 1 if XY are 0
    normals = deprocess(normals) #go back to 0;1


    roughness = mixedMaterial[:,2]
    InvertBoolValue = (boolValue - 1.0) * -1.0
    additionRoughness = tf.tile(tf.clip_by_value(tf.random_normal([1], 0.65, 0.05), 0.0, 1.0), [3]) * InvertBoolValue
    roughness = (roughness * boolValue) + additionRoughness

    mixedMaterial = tf.concat([tf.expand_dims(normals, axis=1), tf.expand_dims(mixedMaterial[:,1], axis = 1), tf.expand_dims(roughness, axis = 1), tf.expand_dims(mixedMaterial[:,3], axis = 1)], axis=1)
    print("adapting normal and roughness")
    return mixedMaterial

#Crops an image using the given parameters. Used to crop the inputs if it was bigger than wanted.
def cropImage(params):
    image = params[0] # shape [height, width, 3]
    topLeftCorner = params[1] # shape [2]
    cropSize = params[2] #shape [1]
    return image[topLeftCorner[0]:topLeftCorner[0] + cropSize, topLeftCorner[1]:topLeftCorner[1] + cropSize, :]

#Crops the input picture to have a proper size input. Allows to have jittering images
def cutSidesOut(inputs, targetstoRender, topLeft, cropSize, firstAsGuide, averageCrop):
    #Inputs have shape [BatchSize, nbRenderings, height, width, 3]
    #Targets gave shape [batchsize, height, width, 3]
    #topleft has shape [batchSize, nbRenderings, 2]
    inputOriginalShape = tf.shape(inputs)
    topLeftOriginalShape = tf.shape(topLeft)

    inputs = tf.reshape(inputs, [-1, inputOriginalShape[2], inputOriginalShape[3], inputOriginalShape[4]])
    topLeftInputs = tf.reshape(topLeft, [-1, topLeftOriginalShape[2]])
    cropsizeTiled = tf.tile(tf.constant([cropSize]), [tf.shape(topLeftInputs)[0]])

    croppedInputs = tf.map_fn(cropImage, (inputs, topLeftInputs, cropsizeTiled), dtype = tf.float32)

    newInputShape = tf.shape(croppedInputs)
    croppedInputs = tf.reshape(croppedInputs, [inputOriginalShape[0], inputOriginalShape[1], newInputShape[1], newInputShape[2], newInputShape[3]])

    cropsizeTiled = tf.tile(tf.constant([cropSize]), [tf.shape(targetstoRender)[0]])
    positionToCropTarget = topLeft[:,0,:]

    if firstAsGuide:
        positionToCropTarget = tf.zeros(tf.shape(positionToCropTarget), dtype=tf.int32) + averageCrop
    croppedTargets = tf.map_fn(cropImage, [targetstoRender, positionToCropTarget, cropsizeTiled], dtype=tf.float32)

    return croppedInputs, croppedTargets


#Materials have shape [batch, targetNb,256,256,3]
#Mix the materials, am particularly careful on the normals as the are vector fields and we don't want to break them.
def mixMaterials(material1, material2, alpha):
    normal1Corrected = (material1[:, 0] - 0.5) * 2.0 #go between -1 and 1
    normal2Corrected = (material2[:, 0] - 0.5) * 2.0
    normal1Projected = normal1Corrected / tf.expand_dims(tf.maximum(0.01, normal1Corrected[:,:,:,2]), axis = -1) #Project the normals to use the X and Y derivative
    normal2Projected = normal2Corrected / tf.expand_dims(tf.maximum(0.01, normal2Corrected[:,:,:,2]), axis = -1)

    mixedNormals = alpha * normal1Projected  + (1.0 - alpha) * normal2Projected
    normalizedNormals = mixedNormals / tf.sqrt(tf.reduce_sum(tf.square(mixedNormals), axis=-1, keep_dims=True))
    normals = (normalizedNormals * 0.5) + 0.5 # Back to 0;1

    mixedRest = alpha * material1[:, 1:] + (1.0 - alpha) * material2[:, 1:]

    final = tf.concat([tf.expand_dims(normals, axis = 1), mixedRest], axis = 1)

    return final

# Generate an array grid between -1;1 to act as the "coordconv" input layer (see coordconv paper)
def generateCoords(inputShape):
    crop_size = inputShape[-2]
    firstDim = inputShape[0]

    Xcoords= tf.expand_dims(tf.lin_space(-1.0, 1.0, crop_size), axis=0)
    Xcoords = tf.tile(Xcoords,[crop_size, 1])
    Ycoords = -1 * tf.transpose(Xcoords) #put -1 in the bottom of the table
    Xcoords = tf.expand_dims(Xcoords, axis = -1)
    Ycoords = tf.expand_dims(Ycoords, axis = -1)
    coords = tf.concat([Xcoords, Ycoords], axis=-1)
    coords = tf.expand_dims(coords, axis = 0)#Add dimension to support batch size and nbRenderings should now be [1, 256, 256, 2].
    coords = tf.tile(coords, [firstDim, 1, 1, 1]) #Add the proper dimension here for concat
    return coords

# Generate an array grid between -1;1 to act as each pixel position for the rendering.
def generateSurfaceArray(crop_size, pixelsToAdd = 0):
    totalSize = crop_size + (pixelsToAdd * 2)
    surfaceArray=[]
    XsurfaceArray = tf.expand_dims(tf.lin_space(-1.0, 1.0, totalSize), axis=0)
    XsurfaceArray = tf.tile(XsurfaceArray,[totalSize, 1])
    YsurfaceArray = -1 * tf.transpose(XsurfaceArray) #put -1 in the bottom of the table
    XsurfaceArray = tf.expand_dims(XsurfaceArray, axis = -1)
    YsurfaceArray = tf.expand_dims(YsurfaceArray, axis = -1)

    surfaceArray = tf.concat([XsurfaceArray, YsurfaceArray, tf.zeros([totalSize, totalSize,1], dtype=tf.float32)], axis=-1)
    surfaceArray = tf.expand_dims(tf.expand_dims(surfaceArray, axis = 0), axis = 0)#Add dimension to support batch size and nbRenderings
    return surfaceArray

#create small variation to be added to the positions of lights or camera.
def jitterPosAround(batchSize, nbRenderings, posTensor, mean = 0.0, stddev = 0.03):
    randomPerturbation =  tf.clip_by_value(tf.random_normal([batchSize, nbRenderings,1,1,1,3], mean, stddev, dtype=tf.float32), -0.24, 0.24) #Clip here how far it can go to 8 * stddev to avoid negative values on view or light ( Z minimum value is 0.3)
    return posTensor + randomPerturbation

#Generate input renderings with a light and view setup depending on what was asked by the user. Edit this function and acquisitionScene.py to add more options for the input image rendering.
def generateInputRenderings(rendererInstance, material, batchSize, nbRenderings, surfaceArray, renderingScene, jitterLightPos, jitterViewPos, useAmbientLight, useAugmentationInRenderings = True):
    currentLightPos, currentViewPos, currentConeTargetPos = None, None, None
    if renderingScene == "staticViewPlaneLight":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.defaultScene(surfaceArray, batchSize, nbRenderings)
    elif renderingScene == "staticViewSpotLight":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.defaultSceneSpotLight(surfaceArray, batchSize, nbRenderings)
    elif renderingScene == "staticViewHemiSpotLight":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.fixedViewHemisphereConeLight(surfaceArray, batchSize, nbRenderings)
    elif renderingScene == "staticViewHemiSpotLightOneSurface":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.fixedViewHemisphereConeLightOneOnPlane(surfaceArray, batchSize, nbRenderings)
    elif renderingScene == "movingViewHemiSpotLightOneSurface":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.movingViewHemisphereConeLightOneOnPlane(surfaceArray, batchSize, nbRenderings, useAugmentationInRenderings)
    elif renderingScene == "fixedAngles":
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.fixedAngles(surfaceArray, batchSize, nbRenderings, useAugmentationInRenderings)
    elif renderingScene == "globalTestScene":
        print("BEWARE, USING TEST SCENE, NEED TO HAVE UNEVEN NBRENDERINGS AND NBRENDERINGS -1 /3 SHOULD BE INT")
        currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation = acquisitionScene.globalTestScene(surfaceArray, batchSize, nbRenderings, useAugmentationInRenderings)
    else:
        raise ValueError("Rendering scene unknown")

    if not useAugmentationInRenderings:
        useAugmentation = False

    print("use Augmentation ? : " + str(useAugmentation))
    if jitterLightPos and not currentLightPos is None:
        currentLightPos = jitterPosAround(batchSize, nbRenderings, currentLightPos, 0.0, 0.03)
    if jitterViewPos and not currentViewPos is None:
        currentViewPos = jitterPosAround(batchSize, nbRenderings, currentViewPos, 0.0, 0.03)
    wo = currentViewPos - surfaceArray
    if useAmbientLight:
        ambientLightPos = acquisitionScene.generateAmbientLight(currentLightPos, batchSize)
        wiAmbient = ambientLightPos - surfaceArray
        renderingAmbient = rendererInstance.tf_Render(material, wiAmbient, wo, None, multiLight = True, currentLightPos = ambientLightPos, lossRendering = False, isAmbient = True, useAugmentation = useAugmentation)[0]
    wi = currentLightPos - surfaceArray

    #tf.summary.image("wi", convert(deprocess(wi[:, 0,0])), max_outputs=5)

    renderings = rendererInstance.tf_Render(material,wi,wo, currentConeTargetPos, multiLight = True, currentLightPos = currentLightPos, lossRendering = False, isAmbient = False, useAugmentation = useAugmentation)[0] # was currentConeTargetPos not None
    if useAmbientLight:
        renderings = renderings + renderingAmbient #Add ambient if necessary
        renderingAmbient = tf.clip_by_value(renderingAmbient, 0.0, 1.0)
        renderingAmbient = tf.pow(renderingAmbient, 0.4545)
        renderingAmbient = tf.image.convert_image_dtype(convert(renderingAmbient), dtype=tf.float32)
        tf.summary.image("renderingAmbient", convert(renderingAmbient[0, :]), max_outputs=5)
    if useAugmentation:
        renderings = addNoise(renderings)
    renderings = tf.clip_by_value(renderings, 0.0, 1.0) # Make sure noise doesn't put values below 0 and simulate over exposure
    renderings = tf.pow(renderings, 0.4545) #gamma the results
    #renderings = tf.clip_by_value(renderings, 0.0, 1.0) # to simulate over exposure
    #renderings = tf.Print(renderings, [tf.reduce_mean(renderings)],  message="mean of renderings: ", summarize=20)
    #renderings = tf.Print(renderings, [tf.reduce_max(renderings)],  message="max of renderings: ", summarize=20)
    renderings = tf.image.convert_image_dtype(convert(renderings), dtype=tf.float32)
    return renderings

#Adds a little bit of noise
def addNoise(renderings):
    shape = tf.shape(renderings)
    stddevNoise = tf.exp(tf.random_normal((), mean = np.log(0.005), stddev=0.3))
    noise = tf.random_normal(shape, mean=0.0, stddev=stddevNoise)
    return renderings + noise

#generate the diffuse rendering for the loss computation
def tf_generateDiffuseRendering(batchSize, nbRenderings, targets, outputs, renderer):
    currentViewPos = tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.1)
    currentLightPos = tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.1)

    wi = currentLightPos
    wi = tf.expand_dims(wi, axis=2)
    wi = tf.expand_dims(wi, axis=2)

    wo = currentViewPos
    wo = tf.expand_dims(wo, axis=2)
    wo = tf.expand_dims(wo, axis=2)

    #Add a dimension to compensate for the nb of renderings
    #targets = tf.expand_dims(targets, axis=-2)
    #outputs = tf.expand_dims(outputs, axis=-2)

    #Here we have wi and wo with shape [batchSize, height,width, nbRenderings, 3]
    renderedDiffuse = renderer.tf_Render(targets,wi,wo, None, "diffuse", useAugmentation = False, lossRendering = True)[0]

    renderedDiffuseOutputs = renderer.tf_Render(outputs,wi,wo, None, "", useAugmentation = False, lossRendering = True)[0]#tf_Render_Optis(outputs,wi,wo)
    #renderedDiffuse = tf.Print(renderedDiffuse, [tf.shape(renderedDiffuse)],  message="This is renderings targets Diffuse: ", summarize=20)
    #renderedDiffuseOutputs = tf.Print(renderedDiffuseOutputs, [tf.shape(renderedDiffuseOutputs)],  message="This is renderings outputs Diffuse: ", summarize=20)
    return [renderedDiffuse, renderedDiffuseOutputs]

#generate the specular rendering for the loss computation
def tf_generateSpecularRendering(batchSize, nbRenderings, surfaceArray, targets, outputs, renderer):
    currentViewDir = tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = 0.001, highEps = 0.1)
    currentLightDir = currentViewDir * tf.expand_dims([-1.0, -1.0, 1.0], axis = 0)
    #Shift position to have highlight elsewhere than in the center.
    currentShift = tf.concat([tf.random_uniform([batchSize, nbRenderings, 2], -1.0, 1.0), tf.zeros([batchSize, nbRenderings, 1], dtype=tf.float32) + 0.0001], axis=-1)

    currentViewPos = tf.multiply(currentViewDir, tf_generate_distance(batchSize, nbRenderings)) + currentShift
    currentLightPos = tf.multiply(currentLightDir, tf_generate_distance(batchSize, nbRenderings)) + currentShift

    currentViewPos = tf.expand_dims(currentViewPos, axis=2)
    currentViewPos = tf.expand_dims(currentViewPos, axis=2)

    currentLightPos = tf.expand_dims(currentLightPos, axis=2)
    currentLightPos = tf.expand_dims(currentLightPos, axis=2)

    wo = currentViewPos - surfaceArray
    wi = currentLightPos - surfaceArray

    #targets = tf.expand_dims(targets, axis=-2)
    #outputs = tf.expand_dims(outputs, axis=-2)
    #targets = tf.Print(targets, [tf.shape(targets)],  message="This is targets in specu renderings: ", summarize=20)
    renderedSpecular = renderer.tf_Render(targets,wi,wo, None, "specu", useAugmentation = False, lossRendering = True)[0]
    renderedSpecularOutputs = renderer.tf_Render(outputs,wi,wo, None, "", useAugmentation = False, lossRendering = True)[0]
    #tf_Render_Optis(outputs,wi,wo, includeDiffuse = a.includeDiffuse)

    #renderedSpecularOutputs = tf.Print(renderedSpecularOutputs, [tf.shape(renderedSpecularOutputs)],  message="This is renderings outputs Specular: ", summarize=20)
    return [renderedSpecular, renderedSpecularOutputs]

#Define a saver that could load only variable which are saved and ignore variables it doesn't have. /!\USE WITH CAUTION/!\ it wil not tell you if nothing was found or loaded.
def optimistic_saver(save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    return tf.train.Saver(restore_vars)

#Check if an event should happen depending of the step of training.
def should(freq, max_steps, step):
    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)


#Reshape the inputs to ba able to process them easily with multiple renderings. Put everything in the first dimension.
def input_reshape(inputs, NoMaxPooling, nbRenderings):
    #take care of reshaping the inputs
    batchSize = tf.shape(inputs)[0]
    if NoMaxPooling:
        renderings = tf.split(inputs, num_or_size_splits=nbRenderings, axis=1)
        reshaped = tf.concat(renderings, axis = -1)
        reshaped = tf.squeeze(reshaped, axis = 1) #this should remove the now useless dim
    else:

        nbInputs = tf.shape(inputs)[1]
        reshaped = tf.reshape(inputs, [batchSize * nbInputs, int(inputs.get_shape()[2]), int(inputs.get_shape()[3]), int(inputs.get_shape()[4])]) #Should be [batchSize * nbInputs, 256,256,3]
    return reshaped, batchSize

# def process_targets(targets):
    # diffuse = targets[:,:,:,3:6]
    # normals = targets[:,:,:,0:3]
    # roughness = targets[:,:,:,6:9]
    # specular = targets[:,:,:,9:12]
    # return tf.concat([normals[:,:,:,0:2],diffuse, tf.expand_dims(roughness[:,:,:,0], axis=-1), specular], axis=-1)

#Put the normals and roughness back to 3 channel for easier processing.
def deprocess_outputs(outputs):
    partialOutputedNormals = outputs[:,:,:,0:2] * 3.0 #The multiplication here gives space to generate direction with angle > pi/4
    outputedDiffuse = outputs[:,:,:,2:5]
    outputedRoughness = outputs[:,:,:,5]
    outputedSpecular = outputs[:,:,:,6:9]
    normalShape = tf.shape(partialOutputedNormals)
    newShape = [normalShape[0], normalShape[1], normalShape[2], 1]
    #normalShape[-1] = 1
    tmpNormals = tf.ones(newShape, tf.float32)

    normNormals = tf_Normalize(tf.concat([partialOutputedNormals, tmpNormals], axis = -1))
    outputedRoughnessExpanded = tf.expand_dims(outputedRoughness, axis = -1)
    return tf.concat([normNormals, outputedDiffuse, outputedRoughnessExpanded, outputedRoughnessExpanded, outputedRoughnessExpanded, outputedSpecular], axis=-1)

#Reshape tensor and add gamma for vizualisation purposes.
def reshape_tensor_display(tensor, splitAmount, logAlbedo = False, axisToSplit = 3):
    tensors_list = tf.split(tensor, splitAmount, axis=axisToSplit)#4 * [batch, 256,256,3]
    if tensors_list[0].get_shape()[1] == 1:
        tensors_list = [tf.squeeze (tensor, axis = 1) for tensor in tensors_list]

    if logAlbedo:
        tensors_list[-1] = logTensor(tensors_list[-1])
        tensors_list[1] = logTensor(tensors_list[1])

    tensors = tf.stack(tensors_list, axis = 1) #[batch, 4,256,256,3]
    shape = tf.shape(tensors)
    newShape = tf.concat([[shape[0] * shape[1]], shape[2:]], axis=0)
    tensors_reshaped = tf.reshape(tensors, newShape)

    return tensors_reshaped

#Register renderings, inputs, targets, outputs and loss value to tensorboard
def registerTensorboard(paths, images, nbInputsMax, nbTargets, loss_value, batch_size, targetsRenderings, outputsRenderings):
        inputs = images[0]
        targets = images[1]
        outputs = images[2]

        targetsList = tf.split(targets, batch_size, axis = 0)
        inputsList = tf.split(inputs, batch_size, axis = 0)
        #print(targetsList[0])
        #inputsList[0] = tf.Print(inputsList[0], [tf.reduce_mean(inputsList[0])], "mean of inputs 0")
        tf.summary.image("targets", targetsList[0], max_outputs=nbTargets)
        tf.summary.image("inputs", inputsList[0], max_outputs=nbInputsMax)
        tf.summary.image("outputs", outputs, max_outputs=nbTargets)
        tf.summary.scalar("loss", loss_value)
        #targetsRenderings is [batchSize,nbRenderings, 256, 256, 3]
        tf.summary.image("targets renderings", tf.unstack(tf.log(targetsRenderings[0] + 0.1), axis=0), max_outputs=9)
        tf.summary.image("outputs renderings", tf.unstack(tf.log(outputsRenderings[0] + 0.1), axis=0), max_outputs=9)

#Deprocess an image to be visible our of tensorflow.
def deprocess_images(inputs, targets, outputs, gammaCorrectedInputs, nbTargets, logAlbedo):
    inputs = deprocess(inputs)
    targets = deprocess(targets)
    outputs = deprocess(outputs)
    #gammaCorrectedInputs = deprocess(gammaCorrectedInputs)
    #inputs = logTensor(inputs)

    with tf.name_scope("transform_images"):
        targetShape = targets.get_shape()
        targets_reshaped = reshape_tensor_display(targets, nbTargets, logAlbedo, axisToSplit = 1)#tf.reshape(targets, [-1, int(targetShape[2]), int(targetShape[3]), int(targetShape[4])])

        outputs_reshaped = reshape_tensor_display(outputs, nbTargets, logAlbedo, axisToSplit = 3)
        inputs_reshaped = tf.reshape(inputs, [-1,tf.shape(inputs)[2],tf.shape(inputs)[3],tf.shape(inputs)[4]])
        gammaCorrectedInputs_reshaped = tf.reshape(gammaCorrectedInputs, [-1, tf.shape(gammaCorrectedInputs)[2], tf.shape(gammaCorrectedInputs)[3], tf.shape(gammaCorrectedInputs)[4]])
    with tf.name_scope("convert_images"):
        converted_inputs = convert(inputs_reshaped)
        converted_targets = convert(targets_reshaped)
        converted_outputs = convert(outputs_reshaped)
        converted_gammaCorrectedInputs = convert(gammaCorrectedInputs_reshaped)
        #converted_inputs = tf.Print(converted_inputs, [tf.shape(converted_inputs)],  message="This is converted_inputs: ", summarize=20)
    return converted_inputs, converted_targets, converted_outputs, converted_gammaCorrectedInputs

#Create the variables to be fetched from tensorflow
def display_images_fetches(paths, inputs, targets, gammaCorrectedInputs, outputs, nbTargets, logAlbedo):

    converted_inputs, converted_targets, converted_outputs, converted_gammaCorrectedInputs  = deprocess_images(inputs, targets, outputs, gammaCorrectedInputs, nbTargets, logAlbedo)
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
            "gammaCorrectedInputs": tf.map_fn(tf.image.encode_png, converted_gammaCorrectedInputs, dtype=tf.string, name="gammaInput_pngs"),
        }
    images = [converted_inputs, converted_targets, converted_outputs]
    return display_fetches, images

#convert the images to uint8
def convert(image, squeeze=False):
    if squeeze:
        def tempLog(imageValue):
            imageValue= tf.log(imageValue + 0.01)
            imageValue = imageValue - tf.reduce_min(imageValue)
            imageValue = imageValue / tf.reduce_max(imageValue)
            return imageValue
        image = [tempLog(imageVal) for imageVal in image]

    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

#Save the images to disk and add the path to a list to be used by the html generator
def save_images(fetches, output_dir, batch_size, nbTargets, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        #fetch inputs

        nbCurrentInput = len(fetches["inputs"])//batch_size # This only works if the nb of rendering is constant in the batch.
        for kind in ["inputs","gammaCorrectedInputs"]:
            fileset[kind] = {}
            for idImage in range(nbCurrentInput):
                fileset[kind][idImage] = save_image(idImage, step, name, kind, image_dir, fetches, nbCurrentInput, i)
        #fetch outputs and targets
        for kind in ["outputs", "targets"]:
            fileset[kind] = {}
            for idImage in range(nbTargets):
                fileset[kind][idImage] = save_image(idImage, step, name, kind, image_dir, fetches, nbTargets, i)
        filesets.append(fileset)
    return filesets

#Save an image to disk
def save_image(idImage, step, name, kind, image_dir, fetches, nbImagesToRead, materialID):
    filename = name + "-" + kind + "-" + str(idImage) + "-.png"
    if step is not None:
        filename = "%08d-%s" % (step, filename)
    out_path = os.path.join(image_dir, filename)
    contents = fetches[kind][materialID * nbImagesToRead + idImage]

    with open(out_path, "wb") as f:
        f.write(contents)
    return filename

#Create the html using the files previously saved on disk.
def append_index(filesets, output_dir, nbTargets, mode, step=False):
    nbMaxInput = 5
    index_path = os.path.join(output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        titles = []
        for nbInput in range(nbMaxInput):
            titles.append(str(nbInput + 1) + "input Outputs")
        titles.append("Ground Truth")
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>Material Name</th>")
        for title in titles:
            index.write("<th>" + title + "</th>")
        index.write("</tr>\n")

    for fileset in filesets:
        for kind in ["gammaCorrectedInputs", "inputs"]:
            index.write("<tr>")
            nbInput = len(fileset[kind])
            if step:
                index.write("<td>%d</td>" % fileset["step"])
            index.write("<td>%s %s</td>" % (fileset["name"], kind))


            for idImage in range(nbInput):
                #filetsetKey = kind + str(idImage)
                index.write("<td><img src='images/%s'></td>" % fileset[kind][idImage])
            if nbInput < nbMaxInput:
                for i in range(nbMaxInput - nbInput):
                    index.write("<td></td>")
            index.write("</tr>")

        maps = ["normal", "diffuse", "roughness", "specular"]
        for idImage in range(nbTargets):
            index.write("<tr>")
            if step:
                index.write("<td>%d</td>" % fileset["step"])
            index.write("<td>%s</td>" % (maps[idImage]))
            for nbSkip in range(nbInput - 1):
                index.write("<td></td>")
            index.write("<td><img src='images/%s'></td>" % fileset["outputs"][idImage])
            if nbInput < nbMaxInput:
                for i in range(nbMaxInput - nbInput):
                    index.write("<td></td>")
            if mode != "eval":
                index.write("<td><img src='images/%s'></td>" % fileset["targets"][idImage])

            index.write("</tr>\n")

    return index_path

#Create a HTML which compiles all the results of a full test (All results with from 1 to N input images) 
def writeGlobalHTML(output_dir, filesets, nbTargets, mode, nbMaxInput):
    index_path = os.path.join(output_dir, "GlobalIndex.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        titles = []
        for nbInput in range(nbMaxInput):
            titles.append(str(nbInput + 1) + "input Outputs")
        titles.append("Ground Truth")
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>Material Name</th>")
        for title in titles:
            index.write("<th>" + title + "</th>")
        index.write("</tr>\n")

    for fileset in filesets:
        for inputType in ["gammaCorrectedInputs", "inputs"]:
            index.write("<tr>")
            index.write("<td>%s %s %s</td>" % (fileset["name"], str(nbMaxInput), inputType))
            for currentInputNb in range(nbMaxInput):
                currentInputDefaultPath = "images/%s" % fileset[inputType][currentInputNb]
                index.write("<td><img src='" + str(nbMaxInput - 1) + "/" + currentInputDefaultPath +"'></td>")

            index.write("</tr>")

        maps = ["normal", "diffuse", "roughness", "specular"]
        for idImage in range(nbTargets):
            index.write("<tr>")
            index.write("<td>%s</td>" % (maps[idImage]))
            currentOutputImage = "images/%s" % fileset["outputs"][idImage]
            for nbInput in range(nbMaxInput):
                index.write("<td><img src='%s/%s'></td>" % (str(nbInput), currentOutputImage))
            if mode != "eval":
                index.write("<td><img src='%s/images/%s'></td>" % (str(nbInput), fileset["targets"][idImage]))

            index.write("</tr>\n")

#Print all trainable variables in the network.
def print_trainable():
    for v in tf.trainable_variables():
        print(str(v.name) + ": " + str(v.get_shape()))
