#!/bin/bash

input_dir="./data/train"
image_count=10
model_dir="./models"
epochs=100

python main.py --mode train --input-dir $input_dir --image-count $image_count --model-dir $model_dir --epochs $epochs --save-frequency 50