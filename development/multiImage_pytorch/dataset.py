import matplotlib.pyplot as plt
import os
import torch
import utils

class SvbrdfDataset(torch.utils.data.Dataset):
    def __init__(self, data_directory, input_image_count, used_input_image_count):
        self.data_directory = data_directory
        self.file_paths = [f for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]

        self.input_image_count = input_image_count
        self.used_input_image_count = used_input_image_count
        #if not self.generate_inputs:


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read full image
        file_path    = self.file_paths[idx]
        full_image   = torch.Tensor(plt.imread(file_path)).permute(2, 0, 1)

        image_parts  = torch.cat(full_image.unsqueeze(0).split(256, dim=-1), 0) # [n, 3, 256, 256]

        input_images = image_parts[(self.input_image_count-self.used_input_image_count):self.input_image_count] # [ni, 3, 256, 256]
        svbrdf       = torch.cat(image_parts[self.input_image_count:].split(1, dim=0), dim=1).squeeze(0)        # [12, 256, 256]

        # Transform to linear RGB
        input_images = utils.gamma_decode(input_images)

        return {'inputs': input_images, 'svbrdf': svbrdf}