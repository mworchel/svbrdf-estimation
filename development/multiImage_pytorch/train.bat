SET input_dir="./data/train"
SET image_count=10
SET model_dir="./models"
SET epochs=100

python main.py --mode train --input-dir %input_dir% --image-count %image_count% --model-dir %model_dir% --epochs %epochs% --save-frequency 50 --retrain