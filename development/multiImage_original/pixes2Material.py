from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import json
import glob
import random
import collections
import math
import time
import dataReader
import model as mod
import losses
import helpers
import shutil
import testHelpers
from random import shuffle

tf.disable_eager_execution()

parser = argparse.ArgumentParser()

if __name__ == '__main__':
    parser.add_argument("--mode", required=True, choices=["train", "test", "export", "eval"])
    parser.add_argument("--output_dir", required=True, help="where to put output files")
else:
    parser.add_argument("--mode", required=False, choices=["train", "test", "export", "eval"])
    parser.add_argument("--output_dir", required=False, help="where to put output files")

parser.add_argument("--input_dir", help="path to xml file containing information images")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=50, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")

parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--test_freq", type=int, default=20000, help="test model every test_freq steps, 0 to disable")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")

parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--input_size", type=int, default=288, help="Size of the input data before cropping to 256x256")

parser.add_argument("--lr", type=float, default=0.00002, help="initial learning rate for adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--nbTargets", type=int, default=1, help="Number of images to output")
parser.add_argument("--nbInputs", type=int, default=1, help="Number of images in the input")


parser.add_argument("--loss", type=str, default="render", choices=["l1", "render", "mixed"], help="Which loss to use instead of the L1 loss")
parser.add_argument("--nbDiffuseRendering", type=int, default=3, help="Number of diffuse renderings in the rendering loss")
parser.add_argument("--nbSpecularRendering", type=int, default=6, help="Number of specular renderings in the rendering loss")

parser.add_argument("--useLog", dest="useLog", action="store_true", help="Use the log for input")
parser.set_defaults(useLog=False)

parser.add_argument("--includeDiffuse", dest="includeDiffuse", action="store_true", help="Include the diffuse term in the specular renderings of the rendering loss ?")
parser.set_defaults(includeDiffuse=False)

parser.add_argument("--logOutputAlbedos", dest="logOutputAlbedos", action="store_true", help="Log the output albedos ? ?")
parser.set_defaults(logOutputAlbedos=False)
parser.add_argument("--poolingtype", type=str, default="max", choices=["max", "mean"], help="Define the type of pooling to use")
parser.add_argument("--imageFormat", type=str, default="png", choices=["jpg", "png", "jpeg", "JPG", "JPEG", "PNG"], help="Which format have the input files")
parser.add_argument("--inputMode", type=str, default="auto", choices=["auto", "xml", "folder", "image"], help="What kind of input to expect")
parser.add_argument("--trainFolder", type=str, default="train", help="train folder to read")
parser.add_argument("--testFolder", type=str, default="test", help="test folder to read")

parser.add_argument("--maxImages", type=int, default=1, help="Maximum number of images for the full test, will run the test with 1 to maxImages")
parser.add_argument("--fixImageNb", dest="fixImageNb", action="store_true", help="Use a fix number of image for the training.")
parser.set_defaults(fixImageNb=False)

parser.add_argument("--feedMethod", type=str, default="render", choices=["files", "render"], help="Which feeding method to use")
parser.add_argument("--renderingScene", type=str, default="staticViewPlaneLight", choices=["staticViewPlaneLight", "staticViewSpotLight", "staticViewHemiSpotLight", "staticViewHemiSpotLightOneSurface", "movingViewHemiSpotLightOneSurface", "fixedAngles", "globalTestScene"], help="Static view with plane light")

parser.add_argument("--jitterLightPos", dest="jitterLightPos", action="store_true", help="Jitter or not the light pos.")
parser.set_defaults(jitterLightPos=False)
parser.add_argument("--jitterViewPos", dest="jitterViewPos", action="store_true", help="Jitter or not the view pos.")
parser.set_defaults(jitterViewPos=False)
parser.add_argument("--useCoordConv", dest="useCoordConv", action="store_true", help="use coordconv in the first convolution slot.")
parser.set_defaults(useCoordConv=False)
parser.add_argument("--useAmbientLight", dest="useAmbientLight", action="store_true", help="use ambient lighting in the rendering.")
parser.set_defaults(useAmbientLight=False)
parser.add_argument("--jitterRenderings", dest="jitterRenderings", action="store_true", help="spatially jitter the renderings.")
parser.set_defaults(jitterRenderings=False)
parser.add_argument("--firstAsGuide", dest="firstAsGuide", action="store_true", help="Use the first picture provided as a guide.")
parser.set_defaults(firstAsGuide=False)
parser.add_argument("--NoMaxPooling", dest="NoMaxPooling", action="store_true", help="Use the max pooling system.")
parser.set_defaults(NoMaxPooling=False)
parser.add_argument("--NoAugmentationInRenderings", dest="NoAugmentationInRenderings", action="store_true", help="Use the max pooling system.")
parser.set_defaults(NoAugmentationInRenderings=False)

#Use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"]="1"
a = parser.parse_args()

#Handles the input mode if on automatic mode
if __name__ == '__main__':
    if a.inputMode == "auto":
        if a.input_dir.lower().endswith(".xml"):
            a.inputMode = "xml"
            print("XML Not supported anymore")
        elif os.path.isdir(a.input_dir):
            a.inputMode = "folder"
        else:
            a.inputMode = "image"

#If we don't use maxpooling we always have to have the same number of input images
if a.NoMaxPooling:
    a.fixImageNb = True

#Size of the input to the network
CROP_SIZE = 256

#Number of features in the last convolution layers of the network (leave the last one to 9 !)
last_convs_chans = [64,32,9]

generateTmpData = False
#Folder in which tmpData will be stored
tmpFolder = "noAugmentationCorrectViewFixedTestSet"

def main():

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    #Load some options from the checkpoint if we provided one.
    loadCheckpointOption()
    #If we feed the network with renderings done in the network for a test run, we save the images before, to be able to compare later with other networks on the same testset.
    if a.mode == "test" and a.feedMethod == "render":
        testHelpers.renderTests(a.input_dir, a.testFolder, a.maxImages, tmpFolder, a.imageFormat, CROP_SIZE, a.nbTargets, a.input_size, a.batch_size, a.renderingScene, a.jitterLightPos, a.jitterViewPos, a.inputMode, a.mode, a.output_dir)
        generateTmpData = True
        a.nbInputs = a.maxImages
        a.feedMethod = "files"
        a.testFolder = tmpFolder
        a.input_size = CROP_SIZE

    backupOutputDir = a.output_dir
    #We run the network once if we a training
    nbRun = 1
    #And as many time as the maximum number of images we want to treat with if testing (to have results with one image, two images, three images etc... to see the improvement)
    if a.mode == "test":
        nbRun = a.maxImages #1
        a.fixImageNb = True
        
    #Now run the network nbRun times.
    for runID in range(nbRun):
        maxInputNb = a.maxImages
        if a.mode == "test":
            maxInputNb = runID + 1 #a.maxImages
            a.output_dir = os.path.join(backupOutputDir, str(runID))
            tf.reset_default_graph()
        
        #Create the output dir if it doesn't exist
        if not os.path.exists(a.output_dir):
            os.makedirs(a.output_dir)

        #Write to the "options" file the different parameters of this run.
        with open(os.path.join(a.output_dir, "options.json"), "w") as f:
            f.write(json.dumps(vars(a), sort_keys=True, indent=4))

        #Create a dataset object
        data = dataReader.dataset(a.input_dir, imageType = a.imageFormat, trainFolder = a.trainFolder, testFolder = a.testFolder, inputNumbers = a.nbInputs, maxInputToRead = maxInputNb, nbTargetsToRead = a.nbTargets, cropSize=CROP_SIZE, inputImageSize=a.input_size, batchSize=a.batch_size, fixCrop = (a.mode == "test"), mixMaterials = (a.mode == "train"), fixImageNb = a.fixImageNb, logInput = a.useLog, useAmbientLight = a.useAmbientLight, jitterRenderings = a.jitterRenderings, firstAsGuide = False, useAugmentationInRenderings = not a.NoAugmentationInRenderings)

        # Populate the list of files the dataset will contain
        data.loadPathList(a.inputMode, a.mode, a.mode == "train")
        
        # Depending on wheter we want to render our input data or directly use files, we create the tensorflow data loading system.        
        if a.feedMethod == "render":
            data.populateInNetworkFeedGraph(a.renderingScene, a.jitterLightPos, a.jitterViewPos, a.mode == "test",  shuffle = a.mode == "train")
        elif a.feedMethod == "files":
            data.populateFeedGraph(shuffle = a.mode == "train")
        
        # Here we reshape the input to have all the images in the first dimension (to treat in parallel)
        inputReshaped, dyn_batch_size = helpers.input_reshape(data.inputBatch, a.NoMaxPooling, a.maxImages)
        
        if a.mode == "train":
            with tf.name_scope("recurrentTest"):
                #Initialize different data for tests.
                dataTest = dataReader.dataset(a.input_dir, imageType = a.imageFormat, testFolder = a.testFolder, inputNumbers = a.nbInputs, maxInputToRead = a.maxImages, nbTargetsToRead = a.nbTargets, cropSize=CROP_SIZE, inputImageSize=a.input_size, batchSize=a.batch_size, fixCrop = True, mixMaterials = False, fixImageNb = a.fixImageNb, logInput = a.useLog, useAmbientLight = a.useAmbientLight, jitterRenderings = a.jitterRenderings, firstAsGuide = a.firstAsGuide, useAugmentationInRenderings = not a.NoAugmentationInRenderings)
                dataTest.loadPathList(a.inputMode, "test", False)
                if a.feedMethod == "render":
                    dataTest.populateInNetworkFeedGraph(a.renderingScene, a.jitterLightPos, a.jitterViewPos, True, shuffle = False)
                elif a.feedMethod == "files":
                    dataTest.populateFeedGraph(False)
                TestinputReshaped, test_dyn_batch_size = helpers.input_reshape(dataTest.inputBatch, a.NoMaxPooling, a.maxImages)
                
        #Reshape the targets to [?(Batchsize), 256,256,12]
        targetsReshaped = helpers.target_reshape(data.targetBatch)

        #Create the object to contain the network model.
        model = mod.Model(inputReshaped, dyn_batch_size, last_convolutions_channels = last_convs_chans, generatorOutputChannels=64, useCoordConv = a.useCoordConv, firstAsGuide = a.firstAsGuide, NoMaxPooling = a.NoMaxPooling, pooling_type=a.poolingtype)
        
        #Initialize the model.
        model.create_model()

        if a.mode == "train":
            #Initialize the regular test network with different data so that it can run regular test sets.
            testTargetsReshaped = helpers.target_reshape(dataTest.targetBatch)
            testmodel = mod.Model(TestinputReshaped, test_dyn_batch_size, last_convolutions_channels = last_convs_chans, generatorOutputChannels=64, reuse_bool=True, useCoordConv = a.useCoordConv, firstAsGuide = a.firstAsGuide, NoMaxPooling = a.NoMaxPooling, pooling_type=a.poolingtype)
            testmodel.create_model()
            
            #Organize the images we want to retrieve from the test network run            
            display_fetches_test, _ = helpers.display_images_fetches(dataTest.pathBatch, dataTest.inputBatch, dataTest.targetBatch, dataTest.gammaCorrectedInputsBatch, testmodel.output, a.nbTargets, a.logOutputAlbedos)
            
            # Compute the training network loss.
            loss = losses.Loss(a.loss, model.output, targetsReshaped, CROP_SIZE, a.batch_size, tf.placeholder(tf.float64, shape=(), name="lr"), a.includeDiffuse)
            loss.createLossGraph()
            
            #Create the training graph part
            loss.createTrainVariablesGraph()


        #Organize the images we want to retrieve from the train network run
        display_fetches, converted_images = helpers.display_images_fetches(data.pathBatch, data.inputBatch, data.targetBatch, data.gammaCorrectedInputsBatch, model.output, a.nbTargets, a.logOutputAlbedos)
        if a.mode == "train":
            #Register inputs, targets, renderings and loss in Tensorboard        
            helpers.registerTensorboard(data.pathBatch, converted_images, a.maxImages, a.nbTargets, loss.lossValue, a.batch_size, loss.targetsRenderings, loss.outputsRenderings)

        #Compute how many paramters the network has
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
            
        #Initialize a saver
        saver = tf.train.Saver(max_to_keep=1)
        if a.checkpoint is not None:
            print("reading model from checkpoint : " + a.checkpoint)
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            partialSaver = helpers.optimistic_saver(checkpoint)
        logdir = a.output_dir if a.summary_freq > 0 else None
        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        #helpers.print_trainable()
        with sv.managed_session() as sess:
            print("parameter_count =", sess.run(parameter_count))
            
            #Loads the checkpoint
            if a.checkpoint is not None:
                print("restoring model from checkpoint : " + a.checkpoint)
                partialSaver.restore(sess, checkpoint)
            
            #Evaluate how many steps to run
            max_steps = 2**32
            if a.max_epochs is not None:
                max_steps = data.stepsPerEpoch * a.max_epochs
            if a.max_steps is not None:
                max_steps = a.max_steps

            #If we want to run a test
            if a.mode == "test" or a.mode == "eval":
                filesets = test(sess, data, max_steps, display_fetches, output_dir = a.output_dir)
                if runID == nbRun - 1 and runID >= 1: #If we are at the last iteration of the test, generate the full html
                    helpers.writeGlobalHTML(backupOutputDir, filesets, a.nbTargets, a.mode, a.maxImages)
            #If we want to train
            if a.mode == "train":
               train(sv, sess, data, max_steps, display_fetches, display_fetches_test, dataTest, saver, loss)



def loadCheckpointOption(mode = a.mode, checkpoint = a.checkpoint):
    if mode == "test" or mode == "eval" or mode == "testfull":
        if checkpoint is None:
            #For testing we absolutely need a checkpoint.
            raise Exception("checkpoint required for test, export or eval mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "nbTargets", "loss", "useLog","useCoordConv", "includeDiffuse", "NoMaxPooling"}
        with open(os.path.join(checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)

    for k, v in a._get_kwargs():
        print(k, "=", v)

def test(sess, data, max_steps, display_fetches, output_dir = a.output_dir):
    #Runs the minimum steps between what is asked by user(max_steps) and how many steps are in the full dataset (stepsPerEpoch)
    sess.run(data.iterator.initializer)
    max_steps = min(data.stepsPerEpoch, max_steps)
    filesets = []
    for step in range(max_steps):
        try:
            #Get the results
            results = sess.run(display_fetches)
            
            #save the output images and add them to the list of outputed items
            filesets.extend(helpers.save_images(results, output_dir, a.batch_size, a.nbTargets))
        except tf.errors.OutOfRangeError:
            print("testing fails in OutOfRangeError")
            continue
    #Create an HTML file to vizualize test results.            
    index_path = helpers.append_index(filesets, output_dir, a.nbTargets, a.mode)
    return filesets

def train(sv, sess, data, max_steps, display_fetches, display_fetches_test, dataTest, saver, loss, output_dir = a.output_dir):
    try:
        # training
        start_time = time.time()
        sess.run(data.iterator.initializer)
        
        #For as many steps as required        
        for step in range(max_steps):
            options = None
            run_metadata = None
            if helpers.should(a.trace_freq, max_steps, step):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            
            #Define the variable to evaluate for tf for any train step.
            fetches = {
                "train": loss.trainOp,
                "global_step": sv.global_step,
            }

            #Add variable to evaluate depending on the current step
            if helpers.should(a.progress_freq, max_steps, step) or step <= 1:
                fetches["loss_value"] = loss.lossValue

            #Add variable to evaluate depending on the current step
            if helpers.should(a.summary_freq, max_steps, step):
                fetches["summary"] = sv.summary_op

            try:
                currentLrValue = a.lr
                if a.checkpoint is None and step < 2000:
                    currentLrValue = step * (0.0005) * a.lr # ramps up to a.lr in the 2000 first iterations to avoid crazy first gradients to have too much impact.

                #Run the network
                results = sess.run(fetches, feed_dict={loss.lr: currentLrValue}, options=options, run_metadata=run_metadata)
            except tf.errors.OutOfRangeError :
                print("training fails in OutOfRangeError, probably a problem with the iterator")
                continue
            
            #Get the current global step from the network results
            global_step = results["global_step"]

            if helpers.should(a.summary_freq, max_steps, step):
                #Add results of rendering to tensorboard is the step is right.
                sv.summary_writer.add_summary(results["summary"], global_step)

            if helpers.should(a.trace_freq, max_steps, step):
                print("recording trace")
                sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % global_step)

            if helpers.should(a.progress_freq, max_steps, step):
                #Print information about the training
                train_epoch = math.ceil(global_step / data.stepsPerEpoch) # global_step will have the correct step count even if we resume from a checkpoint
                train_step = global_step - (train_epoch - 1) * data.stepsPerEpoch
                imagesPerSecond = global_step * a.batch_size / (time.time() - start_time)
                remainingMinutes = ((max_steps - global_step) * a.batch_size)/(imagesPerSecond * 60)
                print("progress  epoch %d  step %d  image/sec %0.1f" % (train_epoch, train_step, imagesPerSecond))
                print("Remaining %0.1f minutes" % (remainingMinutes))
                print("loss_value", results["loss_value"])

            if helpers.should(a.save_freq, max_steps, step):
                #Saves the model of current step.
                print("saving model")
                saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step)
                
            if helpers.should(a.test_freq, max_steps, step) or global_step == 1:
                #Run the test set against the currently training network.
                outputTestDir = os.path.join(a.output_dir, str(global_step))
                test(sess, dataTest, max_steps, display_fetches_test, outputTestDir)
            if sv.should_stop():
                break
    finally:
        #Save everything and run one last test.
        saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step)
        sess.run(data.iterator.initializer)
        outputTestDir = os.path.join(a.output_dir, "final")
        test(sess, dataTest, max_steps, display_fetches_test, outputTestDir )
        #If we generated data, we may want to delete it automatically
        #if generateTmpData:
            #shutil.rmtree(os.path.join(a.input_dir, tmpFolder))

if __name__ == '__main__':
    main()
    
#This runNetwork function is only there to call the network from another python script. Useful for pre and post processing or web server calls.
def runNetwork(inputDir, outputDir, checkpoint, maxImages, inputMode = "image", feedMethod = "files", mode="test", nbInputs=10, input_size=256, nbTargets = 4):
    a.inputMode = inputMode
    a.feedMethod = feedMethod
    a.input_dir = inputDir
    a.output_dir = outputDir
    a.checkpoint = checkpoint
    a.maxImages = maxImages
    a.mode = mode
    a.fixImageNb = True
    a.input_size = input_size
    a.nbInputs = nbInputs
    a.nbTargets=nbTargets
    
    #Print all current options
    print(a)
    #setup all options...
    main()
