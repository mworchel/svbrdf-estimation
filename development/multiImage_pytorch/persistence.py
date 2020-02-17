import gc
import json
import pathlib
import torch

class Checkpoint:
    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint

    @classmethod
    def load(cls, path):
        # # TODO: Handle legacy models
        # checkpoint_path = pathlib.Path(path)
        # checkpoint_dir  = checkpoint_path.parent()
        # state_file_path = checkpoint_dir.joinpath("state.json")
        # if state_file_path.exists():
        return cls(torch.load(path))

    @staticmethod
    def save(path, args, model, optimizer, epoch):
        torch.save({
            'image_size' : args.image_size,
            'model_type' : args.model_type,
            'use_coords' : True if args.use_coords else False,
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    def purge(self):
        self.checkpoint = None
        gc.collect()

    def is_valid(self):
        return self.checkpoint is not None

    def restore_args(self, args):
        # Restore checkpoint relevant arguments
        if self.checkpoint['image_size'] is not None:
            args.image_size = self.checkpoint['image_size']
            print("Restored image size '{}'".format(args.image_size))
        else: 
            print("Failed to restore image size")

        if self.checkpoint['model_type'] is not None:
            args.model_type = self.checkpoint['model_type']
            print("Restored model type '{}'".format(args.model_type))
        else:
            print("Failed to restore model type")

        
        if self.checkpoint['use_coords'] is not None:
            args.use_coords = self.checkpoint['use_coords']
            print("Restored use coords flag '{}'".format(args.use_coords))
        else:
            print("Failed to restore use coords flag")

        return args

    def restore_model_state(self, model):
        if self.checkpoint['model_state_dict'] is not None:
            model.load_state_dict(self.checkpoint['model_state_dict'])
            print("Restored model state")
        else:
            print("Failed to restore model state")

        return model

    def restore_epoch(self, epoch):
        if self.checkpoint['epoch'] is not None:
            epoch = self.checkpoint['epoch']
            print("Restored epoch {}".format(epoch))
        else:
            print("Failed to restore epoch")
        
        return epoch

    def restore_optimizer_state(self, optimizer):
        if self.checkpoint['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            print("Restored optimizer state")
        else:
            print("Failed to restore optimizer state")

        return optimizer

