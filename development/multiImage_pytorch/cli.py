import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SVBRDF Estimation from Images')

    parser.add_argument('--mode', '-M', dest='mode', action='store', required=True,
                        choices=['train', 'test'], default='train',
                        help='Mode in which the script is executed.')

    parser.add_argument('--input-dir', '-i', dest='input_dir', action='store', required=True,
                        help='Directory containing the input data.')

    parser.add_argument('--image-count', '-c', dest='image_count', action='store', required=True,
                        type=int,
                        help='Number of input images (i.e., photographs of the material patch) in the input dataset.')

    parser.add_argument('--used-image-count', '-u', dest='used_image_count', action='store',
                        type=int, default=1,
                        help='Number of input images to use. For the single-view model, values greater than 1 are ignored. When it is greater than image count, the remaining images are artificially generated.')

    parser.add_argument('--image-size', '-s', dest='image_size', action='store',
                        type=int, default=256, 
                        help='Size (width and height) of the image inputs and svbrdf output maps in the model. If the samples in the dataset are larger, they are cropped down to this size.')

    parser.add_argument('--use-coords', dest='use_coords', action='store_true',
                        help='Add spatial image coordinates as features.')

    parser.add_argument('--model-dir', '-m', dest='model_dir', action='store', required=True,
                        help='Directory for the model and training metadata.')

    parser.add_argument('--model-type', dest='model_type', action='store',
                        choices=['single', 'multi'], default='single',
                        help='Which model to use (single-view or multi-view).')  

    parser.add_argument('--save-frequency', dest='save_frequency', action='store', required=False,
                        type=int, choices=range(1, 1000), default=50,
                        metavar="[0-1000]",
                        help='Number of consecutive training epochs after which a checkpoint of the model is saved. Default is %(default)s.')

    parser.add_argument('--epochs', '-e', dest='epochs', action='store',
                        type=int, default=100,
                        help='Maximum number of epochs to run the training for.')

    parser.add_argument('--retrain', dest='retrain', action='store_true',
                        help='When training, ignore any data in the model directory.')

    return parser.parse_args()