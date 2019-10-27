import os
import glob
import tensorflow.compat.v1 as tf
from random import shuffle
import helpers
import math
import renderer
import multiprocessing
import numpy as np

class dataset:
    inputPath = ""
    imageType = ""
    trainFolder = "train"
    testFolder = "test"
    pathList  = []
    imageFormat = "png"
    which_direction = "AtoB"
    inputNumbers = 10
    maxInputToRead = 1
    nbTargetsToRead = 4


    logInput = False
    fixCrop = False
    mixMaterials = True
    fixImageNb = False
    useAmbientLight = False
    jitterRenderings = False
    firstAsGuide = False
    useAugmentationInRenderings = True

    cropSize = 256
    inputImageSize = 288

    maxJitteringPixels = int(np.floor(cropSize / 32.0))
    batchSize = 1
    iterator = None
    inputBatch = None
    targetBatch = None
    pathBatch = None
    gammaCorrectedInputsBatch = None
    stepsPerEpoch = 0


    #Some default constructor with most important parameters
    def __init__(self, inputPath, imageType ="png", trainFolder = "train", testFolder = "test", inputNumbers = 10, maxInputToRead = 1, nbTargetsToRead = 4, cropSize=256, inputImageSize=288, batchSize=1, imageFormat = "png", which_direction = "AtoB", fixCrop = False, mixMaterials = True, fixImageNb = False, logInput = False, useAmbientLight = False, jitterRenderings = False, firstAsGuide = False, useAugmentationInRenderings = True):
        self.inputPath = inputPath
        self.imageType = imageType
        self.trainFolder = trainFolder
        self.testFolder = testFolder
        self.inputNumbers = inputNumbers
        self.maxInputToRead = maxInputToRead
        self.nbTargetsToRead = nbTargetsToRead
        self.cropSize = cropSize
        self.inputImageSize = inputImageSize
        self.batchSize = batchSize
        self.imageFormat = imageFormat
        self.fixCrop = fixCrop
        self.mixMaterials = mixMaterials
        self.fixImageNb = fixImageNb
        self.logInput = logInput
        self.useAmbientLight = useAmbientLight
        self.jitterRenderings = jitterRenderings
        self.firstAsGuide = firstAsGuide
        self.useAugmentationInRenderings = useAugmentationInRenderings

    #Public function to populate the list of path for this dataset
    def loadPathList(self, inputMode, runMode, randomizeOrder):
        if self.inputPath is None or not os.path.exists(self.inputPath):
            raise ValueError("The input path doesn't exist :(!")

        if inputMode == "folder":
            self.__loadFromDirectory(runMode, randomizeOrder)
        if inputMode == "image":
            self.pathList = [self.inputPath]

    #Handles the reading of files from a directory
    def __loadFromDirectory(self, runMode, randomizeOrder = True):
        modeFolder = ""
        if runMode == "train":
            modeFolder = self.trainFolder
        elif runMode == "test":
            modeFolder = self.testFolder

        path = os.path.join(self.inputPath, modeFolder)
        fileList = sorted(glob.glob(path + "/*." + self.imageFormat))
        if randomizeOrder:
            shuffle(fileList)

        if not fileList:
            raise ValueError("The list of filepaths is empty :( : " + path)
        self.pathList = fileList

    #Handles the reading of a single image
    def __readImages(self,filename):
        image_string = tf.read_file(filename) #Gets a string tensor from a file
        decodedInput = tf.image.decode_image(image_string) #Decode a string tensor as image
        floatInput = tf.image.convert_image_dtype(decodedInput, dtype=tf.float32) #Transform image to float32

        assertion = tf.assert_equal(tf.shape(floatInput)[-1], 3, message="image does not have 3 channels")

        with tf.control_dependencies([assertion]):
            floatInput.set_shape([None, None, 3])
            floatInputSplit = tf.split(floatInput, self.nbTargetsToRead + self.inputNumbers, axis=1, name="Split_input_data") #Splitted we get a list of nbTargets + inputNumbers images

        #Sets the inputs and outputs depending on the order of images
        if self.which_direction == "AtoB":
            inputs = floatInputSplit[:self.inputNumbers]
            targets = floatInputSplit[self.inputNumbers:]

        elif self.which_direction == "BtoA":
            inputs = floatInputSplit[self.inputNumbers:]
            targets = floatInputSplit[:self.inputNumbers]
        else:
            raise ValueError("Invalid direction")
        gammadInputs = inputs
        inputs = [tf.pow(input, 2.2) for input in inputs] #correct for the gamma
        #If we want to log the inputs, we do it here
        if self.logInput:
            inputs = [helpers.logTensor(input) for input in inputs]


        #The preprocess function puts the vectors value between [-1; 1] from [0;1]
        inputs = [helpers.preprocess(input) for input in inputs]
        #gammadInputs = [helpers.preprocess(gammadInput) for gammadInput in gammadInputs]
        targets = [helpers.preprocess(target) for target in targets]
        #We used to resize inputs and targets here, we have no functional need for it. Will see if there is a technical need to define the actual size.

        return filename, inputs, targets, gammadInputs

    def __readMaterial(self, material):
        image_string = tf.read_file(material) #Gets a string tensor from a file
        decodedInput = tf.image.decode_image(image_string) #Decode a string tensor as image
        floatMaterial = tf.image.convert_image_dtype(decodedInput, dtype=tf.float32) #Transform image to float32

        return material, tf.split(floatMaterial, self.nbTargetsToRead, axis=1, name="Split_input_data1")

       #Materials are here of shape [batch, nbTargets, 256, 256, 3]
    def __renderInputs(self, materials, renderingScene, jitterLightPos, jitterViewPos, mixMaterials, isTest, renderSize):
        mixedMaterial = materials
        if mixMaterials:
            alpha = tf.random_uniform([1], minval=0.1, maxval=0.9, dtype=tf.float32, name="mixAlpha")
            #print("mat2: " + str(materials2))

            materials1 = materials[::2]
            materials2 = materials[1::2]

            mixedMaterial = helpers.mixMaterials(materials1, materials2, alpha)
        mixedMaterial.set_shape([None, self.nbTargetsToRead, renderSize, renderSize, 3])
        mixedMaterial = helpers.adaptRougness(mixedMaterial)
        #These 3 lines below tries to scale the albedos to get more variety and to randomly flatten the normals to disambiguate the normals and albedos. We did not see strong effect for these.
        #if not isTest and self.useAugmentationInRenderings:
        #    mixedMaterial = helpers.adaptAlbedos(mixedMaterial, self.batchSize)
        #    mixedMaterial = helpers.adaptNormals(mixedMaterial, self.batchSize)

        reshaped_targets_batch = helpers.target_reshape(mixedMaterial) #reshape it to be compatible with the rendering algorithm [?, size, size, 12]
        nbRenderings = self.maxInputToRead
        if not self.fixImageNb:
            #If we don't want a constant number of input images, we randomly select a number of input images between 1 and the maximum number of images defined by the user.
            nbRenderings = tf.random_uniform([1],1, self.maxInputToRead + 1, dtype=tf.int32)[0]
        rendererInstance = renderer.GGXRenderer(includeDiffuse = True)
        ## Do renderings of the mixedMaterial

        targetstoRender = reshaped_targets_batch
        pixelsToAdd = 0

        targetstoRender = helpers.preprocess(targetstoRender) #Put targets to -1; 1
        surfaceArray = helpers.generateSurfaceArray(renderSize, pixelsToAdd) #Generate a grid Y,X between -1;1 to act as the pixel support of the rendering (computer the direction vector between each pixel and the light/view)
        
        #Do the renderings
        inputs = helpers.generateInputRenderings(rendererInstance, targetstoRender, self.batchSize, nbRenderings, surfaceArray, renderingScene, jitterLightPos, jitterViewPos, self.useAmbientLight, useAugmentationInRenderings = self.useAugmentationInRenderings)
        #inputs = [helpers.preprocess(input) for input in inputs]

        randomTopLeftCrop = tf.zeros([self.batchSize, nbRenderings, 2], dtype=tf.int32)
        averageCrop = 0.0
        
        #If we want to jitter the renderings around (to try to take into account small non alignment), we should handle the material crop a bit differently
        #We didn't really manage to get satisfying results with the jittering of renderings. But the code could be useful if this is of interest to Ansys.
        if self.jitterRenderings:
            randomTopLeftCrop = tf.random_normal([self.batchSize, nbRenderings, 2], 0.0, 1.0)#renderSize - self.cropSize, dtype=tf.int32)
            randomTopLeftCrop = randomTopLeftCrop * tf.exp(tf.random_normal([self.batchSize], 0.0, 1.0))#renderSize - self.cropSize, dtype=tf.int32)
            randomTopLeftCrop = randomTopLeftCrop - tf.reduce_mean(randomTopLeftCrop, axis = 1, keep_dims=True)
            randomTopLeftCrop = tf.round(randomTopLeftCrop)
            randomTopLeftCrop = tf.cast(randomTopLeftCrop, dtype=tf.int32)
            averageCrop = tf.cast(self.maxJitteringPixels * 0.5, dtype = tf.int32)
            randomTopLeftCrop = randomTopLeftCrop + averageCrop
            randomTopLeftCrop = tf.clip_by_value(randomTopLeftCrop, 0, self.maxJitteringPixels)

        totalCropSize = self.cropSize
        
        inputs, targets = helpers.cutSidesOut(inputs, targetstoRender, randomTopLeftCrop, totalCropSize, self.firstAsGuide, averageCrop)
        print("inputs shape after" + str(inputs.get_shape()))

        self.gammaCorrectedInputsBatch = inputs
        tf.summary.image("GammadInputs", helpers.convert(inputs[0, :]), max_outputs=5)
        inputs = tf.pow(inputs, 2.2) # correct gamma
        if self.logInput:
            inputs = helpers.logTensor(inputs)

        inputs = helpers.preprocess(inputs)
        targets = helpers.target_deshape(targets, self.nbTargetsToRead)
        return targets, inputs

    def populateInNetworkFeedGraph(self, renderingScene, jitterLightPos, jitterViewPos, isTest, shuffle = True):
        #Create a tensor out of the list of paths
        filenamesTensor = tf.constant(self.pathList)
        #Reads a slice of the tensor, for example, if the tensor is of shape [100,2], the slice shape should be [2] (to check if we have problem here)
        dataset = tf.data.Dataset.from_tensor_slices(filenamesTensor)
        #for each slice apply the __readImages function
        dataset = dataset.map(self.__readMaterial, num_parallel_calls= int(multiprocessing.cpu_count() / 4)) #Divided by four as the cluster divides cpu availiability for each GPU
        #Authorize repetition of the dataset when one epoch is over.
        dataset = dataset.repeat()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=256, reshuffle_each_iteration=True)
        #set batch size
        #print(self.batchSize)
        nbWithdraw = 2 * self.batchSize
        if not self.mixMaterials:
            nbWithdraw = self.batchSize
        batched_dataset = dataset.batch(nbWithdraw)
        batched_dataset = batched_dataset.prefetch(buffer_size=2)
        #batched_dataset = batched_dataset.cache()

        iterator = batched_dataset.make_initializable_iterator()

        #Create the node to retrieve next batch
        init_paths_batch, init_targets_batch = iterator.get_next()
        paths_batch = init_paths_batch
        
        #Get twice more materials on each batch if we need to mix them.
        if self.mixMaterials:
            paths_batch = init_paths_batch[::2]
        renderCropSize = self.cropSize

        #If we want to jitter the renderings a little bit, we do a slightly bigger rendering to be able to crop in it in different location (tries to allow for slightly misaligned inputs).
        if self.jitterRenderings:
            renderCropSize = renderCropSize + self.maxJitteringPixels
        if self.inputImageSize > renderCropSize:
            if self.fixCrop:
                xyCropping = (self.inputImageSize - renderCropSize) // 2
                xyCropping = [xyCropping, xyCropping]
            else:
                xyCropping = tf.random_uniform([1], 0, self.inputImageSize - renderCropSize, dtype=tf.int32)
            init_targets_batch = init_targets_batch[:,:, xyCropping[0] : xyCropping[0] + renderCropSize, xyCropping[0] : xyCropping[0] + renderCropSize, :]
        elif self.inputImageSize < renderCropSize:
            raise Exception("Size of the input is inferior to the size of the rendering, please provide higher resolution maps")

        #Do the renderings
        targets_batch, inputs_batch = self.__renderInputs(init_targets_batch, renderingScene, jitterLightPos, jitterViewPos, self.mixMaterials, isTest, renderCropSize)

        #Set the shapes (to make sure we have the proper dimension as we want). None means that it can be any number (but should have a dimension there), here we use it for the batch size and the number of input pictures that can vary.
        inputs_batch.set_shape([None, None, self.cropSize, self.cropSize, 3])
        targets_batch.set_shape([None, self.nbTargetsToRead, self.cropSize, self.cropSize, 3])

        print("steps per epoch: " + str(int(math.floor(len(self.pathList) / nbWithdraw))))
        #Populate the object
        self.stepsPerEpoch = int(math.floor(len(self.pathList) / nbWithdraw ))
        self.inputBatch = inputs_batch
        self.targetBatch = targets_batch
        self.iterator = iterator
        self.pathBatch = paths_batch

    #This function if used to crate the iterator to go over the data and create the tensors of input and output
    def populateFeedGraph(self, shuffle = True):
        with tf.name_scope("load_images"):
            #Create a tensor out of the list of paths
            filenamesTensor = tf.constant(self.pathList)
            #Reads a slice of the tensor, for example, if the tensor is of shape [100,2], the slice shape should be [2] (to check if we have problem here)
            dataset = tf.data.Dataset.from_tensor_slices(filenamesTensor)

            #for each slice apply the __readImages function
            dataset = dataset.map(self.__readImages, num_parallel_calls=int(multiprocessing.cpu_count() / 4))
            #Authorize repetition of the dataset when one epoch is over.
            dataset = dataset.repeat()
            if shuffle:
               dataset = dataset.shuffle(buffer_size=256, reshuffle_each_iteration=True)
            #set batch size
            batched_dataset = dataset.batch(self.batchSize)
            batched_dataset = batched_dataset.prefetch(buffer_size=4)
            #Create an iterator to be initialized
            iterator = batched_dataset.make_initializable_iterator()

            #Create the node to retrieve next batch
            paths_batch, inputs_batch, targets_batch, gammadInputBatch = iterator.get_next()
            self.gammaCorrectedInputsBatch = gammadInputBatch
            reshaped_targets = helpers.target_reshape(targets_batch)
            randomTopLeftCrop = tf.zeros([self.batchSize, self.inputNumbers, 2], dtype=tf.int32)
            inputRealSize = self.inputImageSize

            targets_batch = helpers.target_deshape(reshaped_targets, self.nbTargetsToRead)
            #Do the random crop, if the crop is fix, crop in the middle
            if inputRealSize > self.cropSize:
                if self.fixCrop:
                    xyCropping = (inputRealSize - self.cropSize) // 2
                    xyCropping = [xyCropping, xyCropping]
                else:
                    xyCropping = tf.random_uniform([1], 0, inputRealSize - self.cropSize, dtype=tf.int32)


                inputs_batch = inputs_batch[:,:, xyCropping[0] : xyCropping[0] + self.cropSize, xyCropping[0] : xyCropping[0] + self.cropSize, :]
                targets_batch = targets_batch[:,:, xyCropping[0] : xyCropping[0] + self.cropSize, xyCropping[0] : xyCropping[0] + self.cropSize, :]

            #Figure out how many inputs should be read and if it should be a random amount
            if self.fixImageNb and self.maxInputToRead > 0:
                nbInputToUse = [self.maxInputToRead]
            else:
                nbInputToUse =  tf.random_uniform([1], minval=1, maxval=(self.maxInputToRead + 1), dtype=tf.int32)

            inputs_batch = inputs_batch[:,:nbInputToUse[0]]

            #Set shapes
            inputs_batch.set_shape([None, None, self.cropSize, self.cropSize, 3])
            targets_batch.set_shape([None, self.nbTargetsToRead, self.cropSize, self.cropSize, 3])

            #Populate the object
            self.stepsPerEpoch = int(math.floor(len(self.pathList) / self.batchSize))
            self.inputBatch = inputs_batch
            self.targetBatch = targets_batch
            self.iterator = iterator
            self.pathBatch = paths_batch

