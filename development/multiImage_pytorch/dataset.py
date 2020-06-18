import environment as env
from PIL import Image
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import renderers
import torch
import utils

class SvbrdfDataset(torch.utils.data.Dataset):
    """
    Class representing a collection of SVBRDF samples with corresponding input images (rendered or real views of the SVBRDF)
    """

    def __init__(self, data_directory, image_size, scale_mode, input_image_count, used_input_image_count, use_augmentation, mix_materials=False, no_svbrdf=False, is_linear=False, random_crop=False):
        self.data_directory = data_directory
        self.file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]

        self.image_size             = image_size
        self.scale_mode             = scale_mode
        self.input_image_count      = input_image_count
        self.used_input_image_count = used_input_image_count
        # No augmentation means fixed view distance, light intensity, neutral whitebalance and constant flash falloff
        self.use_augmentation       = use_augmentation

        # Material mixing augments the dataset by mixing together two materials
        self.mix_materials = mix_materials
        if self.mix_materials and self.input_image_count > 0:
            self.mix_materials = False
            print("Warning: Material mixing is only supported for datasets without input images.")

        # The dataset does not contain any svbrdf maps
        self.no_svbrdf = no_svbrdf

        # The images in the dataset are already linear RGB
        self.is_linear = is_linear

        # If scale mode is 'crop', crop out a randomly placed window from the image
        self.random_crop = random_crop

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_images, svbrdf = self.read_sample(self.file_paths[idx])

        if self.mix_materials:
            import random
            other_index     = random.randrange(0, self.__len__())
            _, other_svbrdf = self.read_sample(self.file_paths[other_index])
            svbrdf          = self.mix(svbrdf, other_svbrdf)

        if self.scale_mode == 'resize':
            width  = input_images.shape[-1]
            height = input_images.shape[-2]

            # Crop out the center of the sample
            is_landscape = width > height
            crop_anchor  = torch.IntTensor([0, (width - height) // 2]) if is_landscape else torch.IntTensor([(height - width) // 2, 0])
            crop_size    = height if is_landscape else width

            input_images = utils.crop_square(input_images, crop_anchor, crop_size)
            svbrdf       = utils.crop_square(svbrdf, crop_anchor, crop_size)

            # Scale the images and the SVBRDF down to the desired size
            # Note: Do not use bicubic interpolation, as it might produce negative values or values > 1 which requires clamping
            input_images = torch.nn.functional.interpolate(input_images,        size=(self.image_size, self.image_size), mode='bilinear')
            svbrdf       = torch.nn.functional.interpolate(svbrdf.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear').squeeze(0)
        elif self.scale_mode == 'crop':
            width  = input_images.shape[-1]
            height = input_images.shape[-2]

            # Determine the top left point of the cropped image
            # TODO: For random jittering, the crop size would need to be a little bit larger than the final size 
            #       and a final cropping stage would be needed after image generation, which produces slightly unaligned images.
            crop_anchor = torch.IntTensor([0, 0])
            if self.random_crop:
                crop_anchor = torch.IntTensor([np.random.randint(0, height - self.image_size + 1), np.random.randint(0, width - self.image_size + 1)])

            # Crop down the svbrdf and the given input images
            svbrdf       = utils.crop_square(svbrdf, crop_anchor, self.image_size)
            input_images = utils.crop_square(input_images, crop_anchor, self.image_size) 
        else:
            raise ValueError("Unknown scale mode {}".format(self.scale_mode))

        # Transform to linear RGB
        if not self.is_linear:
            input_images = utils.gamma_decode(input_images)

        # Images which cannot be read must be generated artificially
        generated_input_image_count = self.used_input_image_count - input_images.shape[0]
        if generated_input_image_count > 0:
            # Constants as defined in the reference code
            min_eps              = 0.001 # Reference: "allows near 90     degrees angles"
            max_eps              = 0.02  # Reference: "removes all angles below 8.13 degrees."
            fixed_light_distance = 2.197
            fixed_view_distance  = 2.75  # Reference: "39.98 degrees FOV"

            # Generate scenes (camera and light configurations)
            # The in the first configuration, the light and view direction are guaranteed to be perpendicular to the material sample.
            # For the remaining cases, both are randomly sampled from a hemisphere.
            light_poses = torch.cat([torch.Tensor(2).uniform_(-0.75, 0.75), torch.ones(1) * fixed_light_distance], dim=-1).unsqueeze(0)
            if generated_input_image_count > 1:
                light_poses_hemisphere = utils.generate_normalized_random_direction(generated_input_image_count - 1, min_eps=min_eps, max_eps=max_eps) * fixed_light_distance
                light_poses            = torch.cat([light_poses, light_poses_hemisphere], dim=0)

            light_colors = torch.Tensor([30.0]).unsqueeze(-1)
            if self.use_augmentation:
                # Reference: "add a normal distribution to the stddev so that sometimes in a minibatch all the images are consistant and sometimes crazy".
                # NOTE: For us, this effect will not be batch-wide but only for this individual sample.
                # FIXME: Since our renderer is differently implemented, the color variations with the given standard deviations
                #        merely have an effect.
                std_deviation = torch.exp(torch.Tensor(1).normal_(mean = -2.0, std = 0.5)).numpy()[0]
                light_colors  = torch.abs(torch.Tensor(generated_input_image_count).normal_(mean = 20.0, std = std_deviation)).unsqueeze(-1)
            light_colors = light_colors.expand(generated_input_image_count, 3)

            # Handle white balance by varying the light color not the camera properties
            if self.use_augmentation:
                white_balance = torch.abs(torch.Tensor(generated_input_image_count, 3).normal_(mean = 1.0, std = 0.03))
                light_colors  = light_colors * white_balance

            if self.use_augmentation:
                # Reference: "Simulates a FOV between 30 degrees and 50 degrees centered around 40 degrees"
                # NOTE: This probably does not do what the reference code expects it to do.
                #       The uniform distribution generates view distances in [0.25, 2.75] which
                #       correspond to FOVs between roughly 150 degrees and 40 degrees.
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
                scene = env.Scene(env.Camera(view_poses[i]), env.Light(light_poses[i], light_colors[i]))
                
                rendering = renderer.render(scene, svbrdf.unsqueeze(0))

                # Simulate noise
                std_deviation_noise = torch.exp(torch.Tensor(1).normal_(mean = np.log(0.005), std=0.3)).numpy()[0]
                noise               = torch.zeros_like(rendering).normal_(mean=0.0, std=std_deviation_noise)
                rendering           = torch.clamp(rendering + noise, min=0.0, max=1.0)

                input_images = torch.cat([input_images, rendering], dim=0)

        # TODO: For random jittering we would need to re-crop to the final size at this point 
        #       with individual crop anchors for all the images.

        return {'inputs': input_images, 'svbrdf': svbrdf}

    def read_sample(self, file_path):
        # Read full image
        # TODO: Use utils.read_image_tensor()
        full_image   = torch.Tensor(plt.imread(file_path)).permute(2, 0, 1)

        # Split the full image apart along the horizontal direction 
        # Magick number 4 is the number of maps in the SVBRDF
        svbrdf_map_count = 0 if self.no_svbrdf else 4 
        image_parts      = torch.cat(full_image.unsqueeze(0).chunk(self.input_image_count + svbrdf_map_count, dim=-1), 0) # [n, 3, 256, 256]

        # Read the SVBRDF (dummy if there is none in the dataset)
        svbrdf = None
        if self.no_svbrdf:
            # If there are no SVBRDFs in the data, there must be images which we can use as size guide.
            width  = image_parts[0].shape[-1]
            height = image_parts[0].shape[-2]

            normals   = torch.cat([torch.zeros((2, height, width)), torch.ones((1, height, width))], dim=0)
            diffuse   = torch.zeros_like(normals)
            roughness = torch.zeros_like(normals)
            specular  = torch.zeros_like(normals)
        else:
            normals   = image_parts[self.input_image_count + 0].unsqueeze(0)
            normals   = utils.decode_from_unit_interval(normals)
            diffuse   = image_parts[self.input_image_count + 1].unsqueeze(0)
            roughness = image_parts[self.input_image_count + 2].unsqueeze(0)
            specular  = image_parts[self.input_image_count + 3].unsqueeze(0)

        svbrdf = utils.pack_svbrdf(normals, diffuse, roughness, specular).squeeze(0) # [12, 256, 256]

        # We read as many input images from the disk as we can
        # FIXME: This is a little bit counter-intuitive, as we are reading the last n images, not the first n
        read_input_image_count = min(self.input_image_count, self.used_input_image_count)
        input_images           = image_parts[(self.input_image_count - read_input_image_count) : self.input_image_count] # [ni, 3, 256, 256]

        return input_images, svbrdf

    def mix(self, svbrdf_0, svbrdf_1, alpha=None):
        if alpha is None:
        alpha = torch.Tensor(1).uniform_(0.1, 0.9)

        normals_0, diffuse_0, roughness_0, specular_0 = utils.unpack_svbrdf(svbrdf_0)
        normals_1, diffuse_1, roughness_1, specular_1 = utils.unpack_svbrdf(svbrdf_1)

        # Reference "Project the normals to use the X and Y derivative"
        normals_0_projected = normals_0 / torch.max(torch.Tensor([0.01]), normals_0[2:3,:,:])
        normals_1_projected = normals_1 / torch.max(torch.Tensor([0.01]), normals_1[2:3,:,:])

        normals_mixed = alpha * normals_0_projected + (1.0 - alpha) * normals_1_projected
        normals_mixed = normals_mixed / torch.sqrt(torch.sum(normals_mixed**2, axis=0,keepdim=True)) # Normalization

        diffuse_mixed   = alpha * diffuse_0 + (1.0 - alpha) * diffuse_1
        roughness_mixed = alpha * roughness_0 + (1.0 - alpha) * roughness_1
        specular_mixed  = alpha * specular_0 + (1.0 - alpha) * specular_1
        
        return utils.pack_svbrdf(normals_mixed, diffuse_mixed, roughness_mixed, specular_mixed)