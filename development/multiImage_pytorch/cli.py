import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SVBRDF Estimation from Images')

    parser.add_argument('--mode', '-M', dest='mode', action='store', required=True,
                        choices=['train', 'test'], default='train',
                        help='Mode in which the script is executed.')

    parser.add_argument('--renderer', '-R', dest='renderer', action='store',
                        choices=['local', 'pathtracing'], default='local',
                        help='Renderer to use for evaluating the rendering loss.')

    parser.add_argument('--input-dir', '-i', dest='input_dir', action='store', required=True,
                        help='Directory containing the input data.')

    parser.add_argument('--image-count', '-c', dest='image_count', action='store', required=True,
                        type=int,
                        help='Number of input images (i.e., photographs of the material patch) in the input dataset.')

    parser.add_argument('--linear-input', dest='linear_input', action='store_true',
                        default=False,
                        help='Flag to indicate that the input images are already in linear RGB.')

    parser.add_argument('--no-svbrdf-input', dest='no_svbrdf_input', action='store_true',
                        default=False,
                        help='Flag to indicate that the input samples do not contain any SVBRDF maps.')

    parser.add_argument('--used-image-count', '-u', dest='used_image_count', action='store',
                        type=int, default=1,
                        help='Number of input images to use. For the single-view model, values greater than 1 are ignored. When it is greater than image count, the remaining images are artificially generated.')

    parser.add_argument('--image-size', '-s', dest='image_size', action='store',
                        type=int, default=256, 
                        help='Size (width and height) of the image inputs and svbrdf output maps in the model. If the samples in the dataset are larger, they are cropped down to this size.')

    parser.add_argument('--scale-mode', dest='scale_mode', action='store',
                        choices=['crop', 'resize'], default='crop',
                        help='Method which is used to make the input samples fit the given image size.')

    parser.add_argument('--use-coords', dest='use_coords', action='store_true',
                        default=False,
                        help='Add spatial image coordinates as features.')

    parser.add_argument('--omit-optimizer-state-save', dest='omit_optimizer_state_save', action='store_true',
                        default=False,
                        help='Do not store the optimizer state in the checkpoint. Setting this option reduces checkpoint size but can impact training continuation negatively.')

    parser.add_argument('--model-dir', '-m', dest='model_dir', action='store', required=True,
                        help='Directory for the model and training metadata.')

    parser.add_argument('--model-type', dest='model_type', action='store',
                        choices=['single', 'multi'], default='single',
                        help='Which model to use (single-view or multi-view).')  

    parser.add_argument('--gpu-id', '-g', dest='gpu_id', action='store', required=False,
                        type=int, default=0,
                        help='Id of the GPU to use. If it is < 0, the CPU is used.')

    parser.add_argument('--save-frequency', dest='save_frequency', action='store', required=False,
                        type=int, choices=range(1, 1000), default=50,
                        metavar="[0-1000]",
                        help='Number of consecutive training epochs after which a checkpoint of the model is saved. Default is %(default)s.')

    parser.add_argument('--epochs', '-e', dest='epochs', action='store',
                        type=int, default=100,
                        help='Maximum number of epochs to run the training for.')

    parser.add_argument('--retrain', dest='retrain', action='store_true',
                        default=False,
                        help='When training, ignore any data in the model directory.')

    args = parser.parse_args()

    # Validate some arguments
    if args.no_svbrdf_input:
        if args.mode=='train':
            raise RuntimeError("Cannot train the model on a samples without SVBRDF maps.")

        if args.image_count == 0:
            raise RuntimeError("No SVBRDF and no image input. What are we supposed to do?")

    return args