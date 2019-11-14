import os
import shutil
import dataReader
import helpers
import tensorflow.compat.v1 as tf
#This file is here to help render datasets that are saved on the disk. It allows to create a random dataset once and for all, before running all trained network on it to evaluate them for example.
#This is mostly a helper for generating a test set and doesn't add to the network itslf.

def concat_tensor_display(tensor, axisToConcat, axisToSplit = 3):
    tensors_list = tf.unstack(tensor, axis=axisToSplit)#4 * [batch, 256,256,3] Might need to use split
    #if tensors_list[0].get_shape()[1] == 1:
    #    tensors_list = [tf.squeeze (tensor, axis = 1) for tensor in tensors_list]

    tensors = tf.concat(tensors_list, axis = axisToConcat) #[batch, 256, 256 * 4, 3]

    return tensors

def concatSplitInputs(inputs, axisToConcat, axisToSplit = 1):
    tensors_list = tf.unstack(inputs, axis=axisToSplit)
    tensorOneInput = tensors_list[0]
    nbImagePerStyle = int((len(tensors_list) - 1 )/ 2)

    SurfaceLightFixedView = tensors_list[1:nbImagePerStyle + 1]
    HemishpereLightFixedView = tensors_list[nbImagePerStyle + 1: (2 * nbImagePerStyle) + 1]
    #HemishpereLightHemisphereView = tensors_list[(2 * nbImagePerStyle) + 1: (3 * nbImagePerStyle) + 1]

    SurfaceLightFixedView = tf.concat(SurfaceLightFixedView, axis = axisToConcat)
    HemishpereLightFixedView = tf.concat(HemishpereLightFixedView, axis = axisToConcat)
    #HemishpereLightHemisphereView = tf.concat(HemishpereLightHemisphereView, axis = axisToConcat)

    return tensorOneInput, SurfaceLightFixedView, HemishpereLightFixedView#, HemishpereLightHemisphereView

def deprocess_images_fullTest(inputs, targets, nbTargets):
    targets = helpers.deprocess(targets)
    with tf.name_scope("transform_images"):
        targetShape = targets.get_shape()
        targets_reshaped = concat_tensor_display(targets, axisToConcat = 2, axisToSplit = 1)
        tensorOneInput, SurfaceLightFixedView, HemishpereLightFixedView  = concatSplitInputs(inputs, axisToConcat = 2, axisToSplit = 1) #HemishpereLightHemisphereView

        tensorOneInput  = tf.concat([tensorOneInput, targets_reshaped], axis = 2)
        SurfaceLightFixedView  = tf.concat([SurfaceLightFixedView, targets_reshaped], axis = 2)
        HemishpereLightFixedView  = tf.concat([HemishpereLightFixedView, targets_reshaped], axis = 2)
        #HemishpereLightHemisphereView  = tf.concat([HemishpereLightHemisphereView, targets_reshaped], axis = 2)

    with tf.name_scope("convert_images"):
        tensorOneInput = helpers.convert(tensorOneInput)
        SurfaceLightFixedView = helpers.convert(SurfaceLightFixedView)
        HemishpereLightFixedView = helpers.convert(HemishpereLightFixedView)
        #HemishpereLightHemisphereView = helpers.convert(HemishpereLightHemisphereView)

    return tensorOneInput, SurfaceLightFixedView, HemishpereLightFixedView#, HemishpereLightHemisphereView

def display_images_fetches_fullTest(paths, inputs, targets, nbTargets):
    tensorOneInput, SurfaceLightFixedView, HemishpereLightFixedView = deprocess_images_fullTest(inputs, targets, nbTargets) #, HemishpereLightHemisphereView
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": paths,
            "tensorOneInput": tf.map_fn(tf.image.encode_png, tensorOneInput, dtype=tf.string, name="tensorOneInput_pngs"),
            "SurfaceLightFixedView": tf.map_fn(tf.image.encode_png, SurfaceLightFixedView, dtype=tf.string, name="SurfaceLightFixedView_pngs"),
            "HemishpereLightFixedView": tf.map_fn(tf.image.encode_png, HemishpereLightFixedView, dtype=tf.string, name="HemishpereLightFixedView_pngs"),
            #"HemishpereLightHemisphereView": tf.map_fn(tf.image.encode_png, HemishpereLightHemisphereView, dtype=tf.string, name="HemishpereLightHemisphereView_pngs"),
        }
    return display_fetches

def deprocess_images(inputs, targets, nbTargets):
    #inputs = helpers.deprocess(inputs)
    targets = helpers.deprocess(targets)

    with tf.name_scope("transform_images"):
        targetShape = targets.get_shape()
        targets_reshaped = concat_tensor_display(targets, axisToConcat = 2, axisToSplit = 1)
        inputs_reshaped = tf.reshape(inputs, [-1,int(inputs.get_shape()[2]),int(inputs.get_shape()[3]),int(inputs.get_shape()[4])])

        tensorToSave  = tf.concat([inputs_reshaped, targets_reshaped], axis = 2)
    with tf.name_scope("convert_images"):
        tensorToSave = helpers.convert(tensorToSave)

    return tensorToSave

def display_images_fetches(paths, inputs, targets, nbTargets):
    tensorToSave = deprocess_images(inputs, targets, nbTargets)
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": paths,
            "tensorToSave": tf.map_fn(tf.image.encode_png, tensorToSave, dtype=tf.string, name="tensorToSave_pngs"),
        }
    return display_fetches

def save_images_fullPath(fetches, output_dir, batch_size, nbTargets, step):
    folders = ["tensorOneInput", "SurfaceLightFixedView", "HemishpereLightFixedView"]#, "HemishpereLightHemisphereView"]
    for folder in folders:
        currentFolder = os.path.join(output_dir, folder)
        if not os.path.exists(currentFolder):
            os.makedirs(currentFolder)
        for i, in_path in enumerate(fetches["paths"]):
            name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
            filename = name + ".png"
            out_path = os.path.join(currentFolder, str(step) + "_" + filename)
            contents = fetches[folder][i]
            with open(out_path, "wb") as f:
               f.write(contents)

def save_images(fetches, output_dir, batch_size, nbTargets):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        filename = name + ".png"
        out_path = os.path.join(output_dir, filename)
        contents = fetches["tensorToSave"][i]
        with open(out_path, "wb") as f:
           f.write(contents)

def renderTests(input_dir, testFolder, maxInputNb, tmpFolder, imageFormat, CROP_SIZE, nbTargets, input_size, batchSize, renderingScene, jitterLightPos, jitterViewPos, inputMode, mode, outputDir):
    fullOutputDir = os.path.join(outputDir, "testGenerationLog")
    if not os.path.exists(fullOutputDir):
        os.makedirs(fullOutputDir)

    fullTmpFolder = os.path.join(input_dir, tmpFolder)
    if os.path.exists(fullTmpFolder):
        shutil.rmtree(fullTmpFolder)

    os.makedirs(fullTmpFolder)

    data = dataReader.dataset(input_dir, imageType = imageFormat, testFolder = testFolder, maxInputToRead = maxInputNb, nbTargetsToRead = nbTargets, cropSize=CROP_SIZE, inputImageSize=input_size, batchSize=batchSize, fixCrop = True, mixMaterials = True, fixImageNb = True, logInput = False, useAmbientLight = False, jitterRenderings = False, useAugmentationInRenderings = False)
    data.loadPathList(inputMode, mode, False)
    data.maxJitteringPixels = 0 #if 0 here, will produce pixel perfect renderings
    data.populateInNetworkFeedGraph(renderingScene, jitterLightPos, jitterViewPos, True, shuffle = True)
    data.gammaCorrectedInputsBatch.set_shape([batchSize, maxInputNb, None, None, None])
    display_fetches = display_images_fetches_fullTest(data.pathBatch, data.gammaCorrectedInputsBatch, data.targetBatch, nbTargets) #display_images_fetches(data.pathBatch, data.gammaCorrectedInputsBatch, data.targetBatch, nbTargets) # save the gamma corrected version of the inputs


    sv = tf.train.Supervisor(logdir=fullOutputDir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        max_steps = 2**32

        sess.run(data.iterator.initializer)
        print(data.stepsPerEpoch)
        max_steps = 100#data.stepsPerEpoch
        for step in range(max_steps):
            try:
                results = sess.run(display_fetches)
                save_images_fullPath(results, fullTmpFolder, batchSize, nbTargets, step)#save_images(results, fullTmpFolder, batchSize, nbTargets)

            except tf.errors.OutOfRangeError:
                print("testing fails in OutOfRangeError")
                continue
    print ("RENDERINGS DONE")
