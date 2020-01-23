import matplotlib.pyplot as plt
import math
import os
import numpy as np
import torch
import utils
import renderers

class SvbrdfDataset(torch.utils.data.Dataset):
    """
    Class representing a collection of SVBRDF samples with corresponding input images (rendered or real views of the SVBRDF)
    """

    def __init__(self, data_directory, image_size, input_image_count, used_input_image_count, use_augmentation):
        self.data_directory = data_directory
        self.file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]

        self.image_size             = image_size
        self.input_image_count      = input_image_count
        self.used_input_image_count = used_input_image_count
        # No augmentation means fixed view distance, light intensity, neutral whitebalance and constant flash falloff
        self.use_augmentation       = use_augmentation

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

        # Query the height of the images, which represents the size of the read images
        # (assuming square images)
        actual_image_size = image_parts.shape[-2]

        # Determine the top left point of the cropped image
        # TODO: If we use jittering, this has to be determined by the random jitter of the rendered images
        crop_anchor = torch.IntTensor([0, 0])

        # Read and crop the SVBRDF
        normals   = image_parts[self.input_image_count + 0].unsqueeze(0)
        normals   = utils.decode_from_unit_interval(normals)
        diffuse   = image_parts[self.input_image_count + 1].unsqueeze(0)
        roughness = image_parts[self.input_image_count + 2].unsqueeze(0)
        specular  = image_parts[self.input_image_count + 3].unsqueeze(0)

        svbrdf = utils.pack_svbrdf(normals, diffuse, roughness, specular).squeeze(0) # [12, 256, 256]

        svbrdf = utils.crop_square(svbrdf, crop_anchor, self.image_size)

        # We read as many input images from the disk as we can and generate the rest artificially
        read_input_image_count      = min(self.input_image_count, self.used_input_image_count)
        generated_input_image_count = self.used_input_image_count - read_input_image_count

        # Use the last of the given input images
        input_images = image_parts[(self.input_image_count - read_input_image_count) : self.input_image_count] # [ni, 3, 256, 256]

        # Crop down the given input images
        input_images = utils.crop_square(input_images, crop_anchor, self.image_size) 

        if generated_input_image_count > 0:
            # Constants as defined in the reference code
            min_eps              = 0.001 # Reference: "allows near 90     degrees angles"
            max_eps              = 0.02  # Reference: "removes all angles below 8.13 degrees."
            fixed_light_distance = 2.197
            fixed_view_distance  = 2.75  # Reference: "39.98 degrees FOV"

            # Generate scenes (camera and light configurations)
            # The in the first configuration, the light and view are guaranteed to be both over the material sample.
            # For the other cases, both are randomly sampled from a hemisphere.
            light_poses = torch.cat([torch.Tensor(2).uniform_(-0.75, 0.75), torch.ones(1) * fixed_light_distance], dim=-1).unsqueeze(0)
            if generated_input_image_count > 1:
                light_poses_hemisphere = utils.generate_normalized_random_direction(generated_input_image_count - 1, min_eps=min_eps, max_eps=max_eps) * fixed_light_distance
                light_poses            = torch.cat([light_poses, light_poses_hemisphere], dim=0)

            light_colors = torch.Tensor([30.0]).unsqueeze(-1)
            if self.use_augmentation:
                # Reference: "add a normal distribution to the stddev so that sometimes in a minibatch all the images are consistant and sometimes crazy".
                # Note: For us, this effect will not be batch-wide but only for this individual sample.
                std_deviation = torch.exp(torch.Tensor(1).normal_(mean = -2.0, std = 0.5)).numpy()[0]
                light_colors  = torch.abs(torch.Tensor(generated_input_image_count).normal_(mean = 20.0, std = std_deviation)).unsqueeze(-1)
            light_colors = light_colors.expand(generated_input_image_count, 3)

            # Handle light balance by varying the light color not the camera properties
            if self.use_augmentation:
                white_balance = torch.abs(torch.Tensor(generated_input_image_count, 3).normal_(mean = 1.0, std = 0.03))
                light_colors  = light_colors * white_balance

            if self.use_augmentation:
                # Reference: "Simulates a FOV between 30 degrees and 50 degrees centered around 40 degrees"
                view_distance = torch.Tensor(generated_input_image_count).uniform_(0.25, 2.75) 
            else:
                view_distance = torch.ones(generated_input_image_count) * fixed_view_distance

            view_poses = torch.cat([torch.Tensor(2).uniform_(-0.25, 0.25), view_distance[:1]], dim=-1).unsqueeze(0)
            if generated_input_image_count > 1:
                view_poses_hemisphere = utils.generate_normalized_random_direction(generated_input_image_count - 1, min_eps=min_eps, max_eps=max_eps) * view_distance[1:].unsqueeze(-1)
                view_poses            = torch.cat([view_poses, view_poses_hemisphere], dim=0)

            renderer = renderers.LocalRenderer()
            for i in range(generated_input_image_count):
                # TODO: Add spotlight support to the renderer (currentConeTargetPos in the reference code)
                scene = renderers.Scene(renderers.Camera(view_poses[i]), renderers.Light(light_poses[i], light_colors[i]))
                
                rendering = renderer.render(scene, svbrdf.unsqueeze(0))

                std_deviation_noise = torch.exp(torch.Tensor(1).normal_(mean = np.log(0.005), std=0.3)).numpy()[0]
                noise = torch.zeros_like(rendering).normal_(mean=0.0, std=std_deviation_noise)
                rendering = torch.clamp(rendering + noise, min=0.0, max=1.0)

                rendering = utils.gamma_encode(rendering)

                input_images = torch.cat([input_images, rendering], dim=0)

        # TODO: For random jittering we need individual crop anchors here
        # input_images = utils.crop_square(input_images, crop_anchor, self.image_size)

        # Transform to linear RGB
        input_images = utils.gamma_decode(input_images)

        return {'inputs': input_images, 'svbrdf': svbrdf}