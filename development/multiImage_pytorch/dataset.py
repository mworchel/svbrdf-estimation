import matplotlib.pyplot as plt
import os
import torch
import utils

class SvbrdfDataset(torch.utils.data.Dataset):
    """
    Class representing a collection of SVBRDF samples with corresponding input images (rendered or real views of the SVBRDF)
    """

    def __init__(self, data_directory, input_image_count, used_input_image_count):
        self.data_directory = data_directory
        self.file_paths = [f for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]

        self.input_image_count      = input_image_count
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

        # Split the full image apart along the horizontal direction
        image_parts  = torch.cat(full_image.unsqueeze(0).split(256, dim=-1), 0) # [n, 3, 256, 256]

        # Use the last of the given input images
        input_images = image_parts[(self.input_image_count - self.used_input_image_count) : self.input_image_count] # [ni, 3, 256, 256]

        # Transform to linear RGB
        input_images = utils.gamma_decode(input_images)

        normals   = image_parts[self.input_image_count + 0].unsqueeze(0)
        diffuse   = image_parts[self.input_image_count + 1].unsqueeze(0)
        roughness = image_parts[self.input_image_count + 2].unsqueeze(0)
        specular  = image_parts[self.input_image_count + 3].unsqueeze(0)

        # Transform the normals from [0, 1] to [-1, 1]
        normals = utils.decode_from_unit_interval(normals)

        svbrdf = utils.pack_svbrdf(normals, diffuse, roughness, specular).squeeze(0) # [12, 256, 256]

        return {'inputs': input_images, 'svbrdf': svbrdf}