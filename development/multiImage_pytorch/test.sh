#!/bin/bash

input_dir="./data/test"
image_count=10
model_dir="./models"

python main.py --mode test --input-dir $input_dir --image-count $image_count --model-dir $model_dir