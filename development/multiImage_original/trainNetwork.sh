#!/bin/bash

outputDir="trainedModel/"
inputDir="myInputDir"
max_steps="400000"
loss="mixed"
renderingScene="movingViewHemiSpotLightOneSurface"
checkpoint="/home/adminVDE/trainedModels/trainingOneImage"

python pixes2Material.py --mode train --output_dir $outputDir --input_dir $inputDir --max_steps $max_steps --summary_freq 1000 --progress_freq 500 --save_freq 10000 --test_freq 20000 --batch_size 8 --input_size 512 --nbTargets 4 --loss $loss --useLog --renderingScene $renderingScene --includeDiffuse --which_direction AtoB --lr 0.00002 --inputMode folder --maxImages 5 --feedMethod render --jitterLightPos --jitterViewPos --useCoordConv --useAmbientLight --logOutputAlbedos
#--checkpoint $checkpoint --NoAugmentationInRenderings
