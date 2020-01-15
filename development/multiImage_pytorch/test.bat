SET input_dir="./data/test"
SET image_count=10
SET model_dir="./models"

python main.py --mode test --input-dir %input_dir% --image-count %image_count% --model-dir %model_dir%