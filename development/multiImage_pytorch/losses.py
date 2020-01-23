import renderers
import torch
import torch.nn as nn
import utils

class SVBRDFL1Loss(nn.Module):
    def forward(self, input, target):
        # Split the SVBRDF into its individual components
        input_normals,  input_diffuse,  input_roughness,  input_specular  = utils.unpack_svbrdf(input)
        target_normals, target_diffuse, target_roughness, target_specular = utils.unpack_svbrdf(target)

        return nn.functional.l1_loss(input_normals, target_normals) + nn.functional.l1_loss(input_diffuse, target_diffuse) + nn.functional.l1_loss(input_roughness, target_roughness) + nn.functional.l1_loss(input_specular, target_specular)

class RenderingLoss(nn.Module):
    def __init__(self):
        super(RenderingLoss, self).__init__()
        
        self.renderer = renderers.LocalRenderer()


    def generate_random_scenes(self, count):
        view_positions  = utils.generate_normalized_random_direction(count, 0.001, 0.1)
        light_positions = utils.generate_normalized_random_direction(count, 0.001, 0.1)

        scenes = []
        for i in range(count):
            c = renderers.Camera(view_positions[i])
            l = renderers.Light(light_positions[i], [50.0, 50.0, 50.0])
            scenes.append(renderers.Scene(c, l))

        return scenes

    def forward(self, input, target):
        batch_size = input.shape[0]
        random_configuration_count   = 3
        specular_configuration_count = 6

        batch_input_renderings = []
        batch_target_renderings = []
        for i in range(batch_size):
            scenes = self.generate_random_scenes(random_configuration_count)
        	# TODO: Generate specular configurations
            input_svbrdf  = input[i]
            target_svbrdf = target[i]
            for scene in scenes:
                batch_input_renderings.append(self.renderer.render(scene, input_svbrdf))
                batch_target_renderings.append(self.renderer.render(scene, target_svbrdf))

        input_renderings  = torch.stack(batch_input_renderings, dim=0)
        target_renderings = torch.stack(batch_target_renderings, dim=0)

        loss = nn.functional.l1_loss(torch.log(input_renderings + 1), torch.log(target_renderings + 1))

        return loss

class MixedLoss(nn.Module):
    def __init__(self, l1_weight = 0.1):
        super(MixedLoss, self).__init__()

        self.l1_weight      = l1_weight
        self.l1_loss        = SVBRDFL1Loss()
        self.rendering_loss = RenderingLoss()

    def forward(self, input, target):
        return self.l1_weight * self.l1_loss(input, target) + self.rendering_loss(input, target)