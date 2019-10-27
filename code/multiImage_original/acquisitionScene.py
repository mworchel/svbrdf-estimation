import os
import tensorflow.compat.v1 as tf
import numpy as np
import math
import helpers


minEps = 0.001 #allows near 90degrees angles
maxEps = 0.02 #removes all angles below 8.13 degrees. see helpers.tf_generate_normalized_random_direction for the equation to calculate it.
lightDistance = 2.197
viewDistance = 2.75 # 39.98 degrees FOV

#A scene with fix view and light moving randomly on a plane over the material (6 point lights to approximate the flash size)
#LightPos return dimensions should be [batchSize, nbRenderings, nbLights, 1, 1, 3]
#ViewPos return dimensions should be [1, 1, 1, 1, 1, 3] as it is constant for all renderings here
#currentConeTargetPos return is None here
def defaultScene(surfaceArray, batchSize, nbRenderings):
    #add dimension for multiple lights
    surfaceArray = tf.expand_dims(surfaceArray, axis=2)
    currentLightPos = fullRandomLightsSurface(batchSize, nbRenderings)
    currentViewPos = fixedView()

    return currentLightPos, currentViewPos, None, surfaceArray, True

#A scene with fix view and spot light moving randomly on a plane over the material (6 point lights to approximate the flash size)
#LightPos return dimensions should be [batchSize, nbRenderings, nbLights, 1, 1, 3]
#ViewPos return dimensions should be [1, 1, 1, 1, 1, 3] as it is constant for all renderings here
#currentConeTargetPos return dimensions should be [batchSize, nbRenderings, 1, 1, 1, 3] as it is constant for all lights here
def defaultSceneSpotLight(surfaceArray, batchSize, nbRenderings):
    currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray = defaultScene(surfaceArray, batchSize, nbRenderings)

    currentConeTargetPos = randomConeLightTarget(batchSize, nbRenderings)

    return currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, True

#A scene with fix view and spot light moving randomly on a hemisphere over the material (6 point lights to approximate the flash size)
#LightPos return dimensions should be [batchSize, nbRenderings, nbLights, 1, 1, 3]
#ViewPos return dimensions should be [1, 1, 1, 1, 1, 3] as it is constant for all renderings here
#currentConeTargetPos return dimensions should be [batchSize, nbRenderings, 1, 1, 1, 3] as it is constant for all lights here
def fixedViewHemisphereConeLight(surfaceArray, batchSize, nbRenderings):
    #add dimension for multiple lights
    surfaceArray = tf.expand_dims(surfaceArray, axis=0)

    currentLightPos = fullRandomLightsHemisphere(batchSize, nbRenderings)
    currentViewPos = fixedView()

    currentConeTargetPos = randomConeLightTarget(batchSize, nbRenderings)

    return currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, True

#A scene with fix view and spot light moving randomly on a hemisphere over the material including one approximately over the center of the material(6 point lights to approximate the flash size)
#LightPos return dimensions should be [batchSize, nbRenderings, nbLights, 1, 1, 3]
#ViewPos return dimensions should be [1, 1, 1, 1, 1, 3] as it is constant for all renderings here
#currentConeTargetPos return dimensions should be [batchSize, nbRenderings, 1, 1, 1, 3] as it is constant for all lights here
def fixedViewHemisphereConeLightOneOnPlane(surfaceArray, batchSize, nbRenderings):
    #add dimension for multiple lights
    surfaceArray = tf.expand_dims(surfaceArray, axis=0)
    currentLightPos = randomLightsHemisphereOneSurface(batchSize, nbRenderings)
    currentViewPos = fixedView()
    currentConeTargetPos = randomConeLightTarget(batchSize, nbRenderings)

    return currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, True

#A scene with moving view and spot light moving randomly on a hemisphere over the material including one approximately over the center of the material(6 point lights to approximate the flash size)
#LightPos return dimensions should be [batchSize, nbRenderings, nbLights, 1, 1, 3]
#ViewPos return dimensions should be [batchSize, nbRenderings, 1, 1, 1, 3] as it is constant for all renderings here
#currentConeTargetPos return dimensions should be [batchSize, nbRenderings, 1, 1, 1, 3] as it is constant for all lights here
def movingViewHemisphereConeLightOneOnPlane(surfaceArray, batchSize, nbRenderings, useAugmentation):
    #add dimension for multiple lights
    surfaceArray = tf.expand_dims(surfaceArray, axis=0)
    currentLightPos = randomLightsHemisphereOneSurface(batchSize, nbRenderings)
    currentViewPos = randomViewHemisphereOneSurface(batchSize, nbRenderings, useAugmentation)
    currentConeTargetPos = randomConeLightTarget(batchSize, nbRenderings)

    return currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation

def fixedAngles(surfaceArray, batchSize, nbRenderings, useAugmentation):
    surfaceArray = tf.expand_dims(surfaceArray, axis=0)
    currentViewPos = fixedView()
    currentLightPos = fixedLightAngles(batchSize, nbRenderings)
    currentConeTargetPos = fixedConeLightTargets(batchSize, nbRenderings)
    return currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation


def globalTestScene(surfaceArray, batchSize, nbRenderings, useAugmentation):
    surfaceArray = tf.expand_dims(surfaceArray, axis=0)
    currentLightPos = testSetLight(batchSize, nbRenderings)
    currentViewPos = testSetView(batchSize, nbRenderings, useAugmentation)
    currentConeTargetPos = testSetConeTarget(batchSize, nbRenderings)

    return currentLightPos, currentViewPos, currentConeTargetPos, surfaceArray, useAugmentation

#Returns 6 point lights with position in circle to simulate a flash.
def simulateFlash(lightPos):
    lightPoses = []
    lightPoses.append(lightPos - [0,-0.025, 0])
    lightPoses.append(lightPos - [0, 0.025, 0])
    lightPoses.append(lightPos - [-0.025, 0, 0])
    lightPoses.append(lightPos - [0.025, 0, 0])
    lightPoses.append(lightPos - [0.0177, 0.0177,0])
    lightPoses.append(lightPos - [-0.0177, -0.0177, 0])

    aggregatedLightPos = lightPos#tf.concat(lightPoses, axis = 2)
    return aggregatedLightPos

def testSetLight(batchSize, nbRenderings):
    nbRenderingEach = int((nbRenderings - 1) / 2)
    currentLightPos = tf.random_uniform([batchSize, 1, 2], -0.05, 0.05, dtype=tf.float32)
    currentLightPos = tf.expand_dims(tf.concat([currentLightPos, tf.ones([batchSize, 1, 1]) * lightDistance], axis = -1), axis=-2)
    currentLightPos = tf.expand_dims(currentLightPos, axis=-2)
    currentLightPos = tf.expand_dims(currentLightPos, axis=-2)

    #surfaceLightPos = fullRandomLightsSurface(batchSize, nbRenderingEach)
    #currentViewPos = fixedView()
    fixedLightPos = fixedLightAngles(batchSize, nbRenderingEach)
    HemiLightPos = randomLightsHemisphereOneSurface(batchSize, nbRenderingEach)

    currentLightPos = tf.concat([currentLightPos, fixedLightPos, HemiLightPos], axis = 1)

    currentLightPos = simulateFlash(currentLightPos)

    return currentLightPos

def testSetView(batchSize, nbRenderings, useAugmentation):
    viewDistanceLocal = viewDistance
    nbRenderingEach = int((nbRenderings - 1) / 2)

    currentViewPos = tf.random_uniform([batchSize, 1, 2], -0.05, 0.05, dtype=tf.float32)
    currentViewPos = tf.concat([currentViewPos, tf.ones([batchSize, 1, 1]) * viewDistanceLocal], axis = -1)
    currentViewPos = tf.expand_dims(currentViewPos, axis=-2)#nbLight, height, width
    currentViewPos = tf.expand_dims(currentViewPos, axis=-2)
    currentViewPos = tf.expand_dims(currentViewPos, axis=-2)
    fixedViewPos = fixedView()
    fixedViewPos = tf.tile(fixedViewPos, [1,nbRenderingEach,1,1,1, 1])
    hemiPos = randomViewHemisphereOneSurface(batchSize, nbRenderingEach, useAugmentation)


    currentViewPos = tf.concat([currentViewPos,fixedViewPos, hemiPos], axis = 1)

    return currentViewPos

def testSetConeTarget(batchSize, nbRenderings):
    nbRenderingEach = int((nbRenderings - 1) / 2)
    currentConeTargetPos0 = tf.random_normal([batchSize, 1, 1, 1, 1, 2], 0.0, 0.01, dtype=tf.float32) #Give a first pos to 0
    currentConeTargetPos0 = tf.concat([currentConeTargetPos0, tf.zeros([batchSize, 1, 1, 1, 1, 1])], axis = -1)

    currentConeTargetPos1 = fixedConeLightTargets(batchSize, nbRenderingEach)
    currentConeTargetPos2 = tf.random_normal([batchSize, nbRenderingEach,1, 1, 1, 2], 0.0, 0.20, dtype=tf.float32)
    currentConeTargetPos2 = tf.concat([currentConeTargetPos2, tf.zeros([batchSize, nbRenderingEach, 1, 1, 1, 1])], axis = -1)


    currentConeTargetPos = tf.concat([currentConeTargetPos0, currentConeTargetPos1, currentConeTargetPos2], axis=1)


    #currentConeTargetPos = tf.expand_dims(currentConeTargetPos, axis=2)
    #currentConeTargetPos = tf.expand_dims(currentConeTargetPos, axis=2)
    #currentConeTargetPos = tf.expand_dims(currentConeTargetPos, axis=2)
    return currentConeTargetPos


def fullRandomLightsHemisphere(batchSize, nbRenderings):
    currentLightPos = tf.expand_dims(helpers.tf_generate_normalized_random_direction(batchSize, nbRenderings, lowEps = minEps, highEps = maxEps) * lightDistance, axis = -2) #getting the pos and adding the multi light dim.

    currentLightPos = simulateFlash(currentLightPos)
    currentLightPos = tf.expand_dims(currentLightPos, axis=-2)
    currentLightPos = tf.expand_dims(currentLightPos, axis=-2)
    return currentLightPos

def randomLightsHemisphereOneSurface(batchSize, nbRenderings):
    currentLightPos = tf.random_uniform([batchSize, 1, 2], -0.75, 0.75, dtype=tf.float32)
    currentLightPos = tf.expand_dims(tf.concat([currentLightPos, tf.ones([batchSize, 1, 1])* lightDistance], axis = -1), axis=-2)

    currentLightPos2 = tf.expand_dims(helpers.tf_generate_normalized_random_direction(batchSize, nbRenderings - 1, lowEps = minEps, highEps = maxEps) * lightDistance, axis = -2) #getting the pos and adding the multi light dim.
    currentLightPos = tf.concat([currentLightPos, currentLightPos2], axis = 1)

    currentLightPos = simulateFlash(currentLightPos)
    currentLightPos = tf.expand_dims(currentLightPos, axis=-2)
    currentLightPos = tf.expand_dims(currentLightPos, axis=-2)
    return currentLightPos

def randomViewHemisphereOneSurface(batchSize, nbRenderings, useAugmentation):
    #viewDistanceLocal = viewDistance
    if useAugmentation:
        viewDistanceLocal = tf.random_uniform((batchSize, nbRenderings, 1), 2.75, 0.25) #Simulates a FOV between 30 degrees and 50 degrees centered around 40 degrees
    else:
        viewDistanceLocal = tf.ones([batchSize, nbRenderings, 1]) * viewDistance # tf.tile(tf.reshape(tf.constant(viewDistance),[1,1,1]), [batchSize, nbRenderings, 1])
    currentViewPos = tf.random_uniform([batchSize, 1, 2], -0.25, 0.25, dtype=tf.float32)
    currentViewPos = tf.concat([currentViewPos, tf.ones([batchSize, 1, 1]) * tf.expand_dims(viewDistanceLocal[:,0,:], axis = 1)], axis = -1)

    currentViewPos2 = helpers.tf_generate_normalized_random_direction(batchSize, nbRenderings - 1, lowEps = minEps, highEps = maxEps) * viewDistanceLocal[:,1:,:]#tf.constant([0.0, 0.0, viewDistanceLocal])
    currentViewPos = tf.concat([currentViewPos, currentViewPos2], axis = 1)
    currentViewPos = currentViewPos

    currentViewPos = tf.expand_dims(currentViewPos, axis=-2)#nbLight, height, width
    currentViewPos = tf.expand_dims(currentViewPos, axis=-2)
    currentViewPos = tf.expand_dims(currentViewPos, axis=-2)
    return currentViewPos

def fullRandomLightsSurface(batchSize, nbRenderings):
    currentLightPos = tf.random_uniform([batchSize, nbRenderings, 2], -1.0, 1.0, dtype=tf.float32)
    currentLightPos = tf.expand_dims(tf.concat([currentLightPos, tf.ones([batchSize, nbRenderings, 1])* lightDistance], axis = -1), axis=-2)

    currentLightPos = simulateFlash(currentLightPos)
    currentLightPos = tf.expand_dims(currentLightPos, axis=-2)
    currentLightPos = tf.expand_dims(currentLightPos, axis=-2)
    return currentLightPos

def fixedView():
    currentViewPos = tf.constant([0.0, 0.0, viewDistance])
    currentViewPos = tf.expand_dims(currentViewPos, axis=0)
    currentViewPos = tf.expand_dims(currentViewPos, axis=0)
    currentViewPos = tf.expand_dims(currentViewPos, axis=0)
    currentViewPos = tf.expand_dims(currentViewPos, axis=0)
    currentViewPos = tf.expand_dims(currentViewPos, axis=0)
    return currentViewPos

def randomConeLightTarget(batchSize, nbRenderings):
    currentConeTargetPos = tf.random_normal([batchSize, nbRenderings, 2], 0.0, 0.25, dtype=tf.float32)
    currentConeTargetPos = tf.concat([currentConeTargetPos, tf.zeros([batchSize, nbRenderings, 1])], axis = -1)

    currentConeTargetPos = tf.expand_dims(currentConeTargetPos, axis=2)
    currentConeTargetPos = tf.expand_dims(currentConeTargetPos, axis=2)
    currentConeTargetPos = tf.expand_dims(currentConeTargetPos, axis=2)
    return currentConeTargetPos

# new "ambient light" should have [batchSize, nbRenderings, nbLights, 1, 1, 3]
def generateAmbientLight(currentLightPos, batchSize): #Maybe use the same cos distribution as in the other direction and pick a distance from a exp normal distribution
    #ambiantXY = tf.random_uniform([batchSize, 1, 1, 1, 1, 2], -20.0, 20.0, dtype=tf.float32) # if the size of the sample is 20 cm, this will be between 2 m left and right, top and bottom from the center of the sample maybe change this to gaussian
    #currentHeight = tf.random_uniform([batchSize, 1, 1, 1, 1, 1], 20, 30) # If the sample is 20 cm, this will put the light at 2.0 to 3m above it (approximately size of ceiling plus minus a table)
    ambiantDir = helpers.tf_generate_normalized_random_direction(batchSize, 1, lowEps = minEps, highEps = 0.2) #Here remove angles below 25 ° from the surface
    ambiantDir = tf.expand_dims(ambiantDir, axis=-2)
    ambiantDir = tf.expand_dims(ambiantDir, axis=-2)
    ambiantDir = tf.expand_dims(ambiantDir, axis=-2)

    ambiantPos = ambiantDir * tf.exp(tf.random_normal((), np.log(30.0), 0.15, tf.float32)) #sample between 1.6 m to 5.4m if the sample is 20cm square
    ambiantPos = tf.tile(ambiantPos, [1, tf.shape(currentLightPos)[1], 1, 1, 1, 1])
    return ambiantPos

def fixedLightAngles(batchSize, nbRenderings):
    yMult = []
    xMult = []
    for i in range(nbRenderings):
        if i % 2:
            yMult.append(1.0)
            xMult.append(1.0)
        else:
            yMult.append(-1.0)
            xMult.append(-1.0)

    factorX = np.linspace(0.0, 0.8, nbRenderings)
    angles = np.linspace(np.pi/8, np.pi/2, nbRenderings, endpoint=True)
    x2y2Max = np.square(np.cos(angles))
    xsquare = x2y2Max * factorX
    ysquare = x2y2Max - xsquare
    x = np.sqrt(xsquare) * xMult
    y = np.sqrt(ysquare) * yMult
    z = np.sqrt(1.0 - np.square(x) - np.square(y))

    coords = np.concatenate([np.expand_dims(x, axis = 1), np.expand_dims(y, axis = 1),np.expand_dims(z, axis = 1)], axis = 1)
    coords = np.expand_dims(coords, axis = 0)# batch
    coords = np.expand_dims(coords, axis = -2)#nbLight
    coords = np.expand_dims(coords, axis = -2)#x
    coords = np.expand_dims(coords, axis = -2)#y

    return tf.constant(coords, dtype=tf.float32)

def fixedConeLightTargets(batchSize, nbRenderings):
    targets = zeroConeLightTarget(batchSize, nbRenderings)
    newTargets = [[0.0, 0.0, 0.0], [-0.5, -0.5, 0.0], [0.4, 0.4, 0.0], [-0.7, 0.1, 0.0], [0.0, 0.0, 0.0]]
    #newTargets = [[0.0, 0.0, 0.0], [-0.3, -0.3, 0.0], [0.2, 0.2, 0.0], [-0.5, 0.1, 0.0], [0.0, 0.0, 0.0]]
    newTargets = tf.expand_dims(newTargets, axis = 0)
    newTargets = tf.expand_dims(newTargets, axis = -2)
    newTargets = tf.expand_dims(newTargets, axis = -2)
    newTargets = tf.expand_dims(newTargets, axis = -2)
    return targets + newTargets

def zeroConeLightTarget(batchSize, nbRenderings):
    return tf.zeros([batchSize, nbRenderings, 1, 1, 1, 3])

