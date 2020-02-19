import gc
import json
import pathlib
import torch

class Checkpoint:
    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint

    @staticmethod
    def get_checkpoint_path(checkpoint_dir):
        return checkpoint_dir.joinpath("checkpoint.tar")

    @staticmethod
    def load_legacy(model_dir):
        model_path = model_dir.joinpath("model.data")
        state_path = model_dir.joinpath("state.json")
        if not model_path.exists():
            return None
        
        checkpoint = {
            'model_state_dict' : torch.load(model_path),
        }
        print("Loaded legacy model state")

        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
                checkpoint['epoch'] = state['epoch']
            print("Loaded legacy training state")

        return checkpoint 

    @classmethod
    def load(cls, checkpoint_dir):
        if not isinstance(checkpoint_dir, pathlib.Path):
            checkpoint_dir = pathlib.Path(checkpoint_dir)
        
        checkpoint_path = Checkpoint.get_checkpoint_path(checkpoint_dir)

        if not checkpoint_path.exists():
            # If there is no checkpoint file we try to perform a legacy load
            checkpoint = Checkpoint.load_legacy(checkpoint_dir)

            if checkpoint is None:
                print("No checkpoint found in directory '{}'".format(checkpoint_dir))

            return cls(checkpoint)

        return cls(torch.load(checkpoint_path))

    @staticmethod
    def save(checkpoint_dir, args, model, optimizer, epoch):
        if not isinstance(checkpoint_dir, pathlib.Path):
            checkpoint_dir = pathlib.Path(checkpoint_dir)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_type' : args.model_type,
            'use_coords' : True if args.use_coords else False,
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
        }

        if not args.omit_optimizer_state_save:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, Checkpoint.get_checkpoint_path(checkpoint_dir))

    def purge(self):
        self.checkpoint = None
        gc.collect()

    def is_valid(self):
        return self.checkpoint is not None

    def restore_args(self, args):
        # Restore checkpoint relevant arguments

        if 'model_type' in self.checkpoint:
            args.model_type = self.checkpoint['model_type']
            print("Restored model type '{}'".format(args.model_type))
        else:
            print("Failed to restore model type")

        
        if 'use_coords' in self.checkpoint:
            args.use_coords = self.checkpoint['use_coords']
            print("Restored use coords flag '{}'".format(args.use_coords))
        else:
            print("Failed to restore use coords flag")

        return args

    def restore_model_state(self, model):
        if 'model_state_dict' in self.checkpoint:
            model.load_state_dict(self.checkpoint['model_state_dict'])
            print("Restored model state")
        else:
            print("Failed to restore model state")

        return model

    def restore_epoch(self, epoch):
        if 'epoch' in self.checkpoint:
            epoch = self.checkpoint['epoch']
            print("Restored epoch {}".format(epoch))
        else:
            print("Failed to restore epoch")
        
        return epoch

    def restore_optimizer_state(self, optimizer):
        if 'optimizer_state_dict' in self.checkpoint:
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            print("Restored optimizer state")
        else:
            print("Failed to restore optimizer state")

        return optimizer

