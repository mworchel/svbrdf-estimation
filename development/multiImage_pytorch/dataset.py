import matplotlib.pyplot as plt
import math
import os
import torch
import utils
import renderers

class SvbrdfDataset(torch.utils.data.Dataset):
    """
    Class representing a collection of SVBRDF samples with corresponding input images (rendered or real views of the SVBRDF)
    """

    def __init__(self, data_directory, input_image_count, used_input_image_count):
        self.data_directory = data_directory
        self.file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]

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
        # Magick number 4 is the number of maps in the SVBRDF
        image_parts  = torch.cat(full_image.unsqueeze(0).chunk(self.input_image_count + 4, dim=-1), 0) # [n, 3, 256, 256]

        # Read the SVBRDF
        normals   = image_parts[self.input_image_count + 0].unsqueeze(0)
        diffuse   = image_parts[self.input_image_count + 1].unsqueeze(0)
        roughness = image_parts[self.input_image_count + 2].unsqueeze(0)
        specular  = image_parts[self.input_image_count + 3].unsqueeze(0)

        # Transform the normals from [0, 1] to [-1, 1]
        normals = utils.decode_from_unit_interval(normals)

        svbrdf = utils.pack_svbrdf(normals, diffuse, roughness, specular).squeeze(0) # [12, 256, 256]

        # We read as many input images from the disk as we can and generate the rest artificially
        read_input_image_count      = min(self.input_image_count, self.used_input_image_count)
        generated_input_image_count = self.used_input_image_count - read_input_image_count

        # Use the last of the given input images
        input_images = image_parts[(self.input_image_count - read_input_image_count) : self.input_image_count] # [ni, 3, 256, 256]

        # TODO: Generate remaining input images by rendering
        # TODO: Choose a random number if we are training and we don't request it to be fix (see fixImageNb for reference)
        if generated_input_image_count > 0:
            renderer = renderers.LocalRenderer()
            for i in range(generated_input_image_count):
                # TODO: Create a (random) scene
                scene = renderers.Scene(renderers.Camera([0.0, 0.0, 2.0]), renderers.Light([0.0, 0.0, 2.0], [30.0, 30.0, 30.0]))
                
                rendering = renderer.render(scene, svbrdf.unsqueeze(0))

                # TODO: Add noise and clamp again

                rendering = utils.gamma_encode(rendering)

                input_images = torch.cat([input_images, rendering], dim=0)


        # Transform to linear RGB
        input_images = utils.gamma_decode(input_images)

        return {'inputs': input_images, 'svbrdf': svbrdf}