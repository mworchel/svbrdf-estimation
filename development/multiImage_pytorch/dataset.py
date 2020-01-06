import matplotlib.pyplot as plt
import math
import os
import torch
import utils
import renderers

def generate_normalized_random_direction(count, min_eps = 0.001, max_eps = 0.05):
    r1 = torch.Tensor(count, 1).uniform_(0.0 + min_eps, 1.0 - max_eps)
    r2 = torch.Tensor(count, 1).uniform_(0.0, 1.0)

    r   = torch.sqrt(r1)
    phi = 2 * math.pi * r2
    
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(1.0 - r**2)

    return torch.cat([x, y, z], axis=-1)

class SvbrdfDataset(torch.utils.data.Dataset):
    """
    Class representing a collection of SVBRDF samples with corresponding input images (rendered or real views of the SVBRDF)
    """

    def __init__(self, data_directory, image_size, input_image_count, used_input_image_count):
        self.data_directory = data_directory
        self.file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]

        self.image_size             = 256
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

        # Query the height of the images, which represents the size of the read images
        # (assuming square images)
        actual_image_size = image_parts.shape[-2]

        # Determine the top left point of the cropped image
        # TODO: If we use jittering, this has to be determined by the random jitter of the rendered images
        crop_anchor = torch.IntTensor([0, 0])

        # Read the SVBRDF
        normals   = image_parts[self.input_image_count + 0].unsqueeze(0)
        diffuse   = image_parts[self.input_image_count + 1].unsqueeze(0)
        roughness = image_parts[self.input_image_count + 2].unsqueeze(0)
        specular  = image_parts[self.input_image_count + 3].unsqueeze(0)

        # Transform the normals from [0, 1] to [-1, 1]
        normals = utils.decode_from_unit_interval(normals)

        svbrdf = utils.pack_svbrdf(normals, diffuse, roughness, specular).squeeze(0) # [12, 256, 256]

        # Crop the SVBRDF
        svbrdf = utils.crop_square(svbrdf, crop_anchor, self.image_size)

        # We read as many input images from the disk as we can and generate the rest artificially
        read_input_image_count      = min(self.input_image_count, self.used_input_image_count)
        generated_input_image_count = self.used_input_image_count - read_input_image_count

        # Use the last of the given input images
        input_images = image_parts[(self.input_image_count - read_input_image_count) : self.input_image_count] # [ni, 3, 256, 256]

        # Crop down the given input images
        input_images = utils.crop_square(input_images, crop_anchor, self.image_size) 

        # TODO: Choose a random number if we are training and we don't request it to be fix (see fixImageNb for reference)
        if generated_input_image_count > 0:
            # Constants as defined in the reference code
            min_eps              = 0.001
            max_eps              = 0.02
            fixed_light_distance = 2.197
            fixed_view_distance  = 2.75 

            # Generate scenes (camera and light configurations)
            light_poses = torch.cat([torch.Tensor(2).uniform_(-0.75, 0.75), torch.ones(1) * fixed_light_distance], dim=-1).unsqueeze(0)
            if generated_input_image_count > 1:
                light_poses_hemisphere = generate_normalized_random_direction(generated_input_image_count - 1, min_eps=min_eps, max_eps=max_eps) * fixed_light_distance
                light_poses            = torch.cat([light_poses, light_poses_hemisphere], dim=0)

            # TODO: Make "augmentation" optional. No augmentation means fixed view distance
            use_augmentation = True
            if use_augmentation:
                view_distance = torch.Tensor(generated_input_image_count).uniform_(0.25, 2.75) # Ref: "Simulates a FOV between 30 degrees and 50 degrees centered around 40 degrees"
            else:
                view_distance = torch.ones(generated_input_image_count) * fixed_view_distance

            view_poses = torch.cat([torch.Tensor(2).uniform_(-0.25, 0.25), view_distance[:1]], dim=-1).unsqueeze(0)
            if generated_input_image_count > 1:
                view_poses_hemisphere = generate_normalized_random_direction(generated_input_image_count - 1, min_eps=min_eps, max_eps=max_eps) * view_distance[1:]
                view_poses            = torch.cat([view_poses, view_poses_hemisphere], dim=0)

            renderer = renderers.LocalRenderer()
            for i in range(generated_input_image_count):
                # TODO: Add random light characteristics
                # FIXME: How to handle cone light target from the reference?
                scene = renderers.Scene(renderers.Camera(view_poses[i]), renderers.Light(light_poses[i], [30.0, 30.0, 30.0]))
                
                rendering = renderer.render(scene, svbrdf.unsqueeze(0))

                # TODO: Add noise and clamp again

                rendering = utils.gamma_encode(rendering)

                input_images = torch.cat([input_images, rendering], dim=0)

        # TODO: For random jittering we need individual crop anchors here
        # input_images = utils.crop_square(input_images, crop_anchor, self.image_size)

        # Transform to linear RGB
        input_images = utils.gamma_decode(input_images)

        return {'inputs': input_images, 'svbrdf': svbrdf}