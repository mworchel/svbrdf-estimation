import environment as env
import renderers
import torch
import torch.nn as nn
import utils

class SVBRDFL1Loss(nn.Module):
    def forward(self, input, target):
        # Split the SVBRDF into its individual components
        input_normals,  input_diffuse,  input_roughness,  input_specular  = utils.unpack_svbrdf(input)
        target_normals, target_diffuse, target_roughness, target_specular = utils.unpack_svbrdf(target)

        epsilon_l1      = 0.01
        input_diffuse   = torch.log(input_diffuse   + epsilon_l1)
        input_specular  = torch.log(input_specular  + epsilon_l1)
        target_diffuse  = torch.log(target_diffuse  + epsilon_l1)
        target_specular = torch.log(target_specular + epsilon_l1)

        return nn.functional.l1_loss(input_normals, target_normals) + nn.functional.l1_loss(input_diffuse, target_diffuse) + nn.functional.l1_loss(input_roughness, target_roughness) + nn.functional.l1_loss(input_specular, target_specular)

class RenderingLoss(nn.Module):
    def __init__(self):
        super(RenderingLoss, self).__init__()
        
        self.renderer = renderers.LocalRenderer()
        self.random_configuration_count   = 3
        self.specular_configuration_count = 6

    def forward(self, input, target):
        batch_size = input.shape[0]

        batch_input_renderings = []
        batch_target_renderings = []
        for i in range(batch_size):
            scenes = env.generate_random_scenes(self.random_configuration_count) + env.generate_specular_scenes(self.specular_configuration_count)
            input_svbrdf  = input[i]
            target_svbrdf = target[i]
            input_renderings  = []
            target_renderings = []
            for scene in scenes:
                input_renderings.append(self.renderer.render(scene, input_svbrdf))
                target_renderings.append(self.renderer.render(scene, target_svbrdf))
            batch_input_renderings.append(torch.cat(input_renderings, dim=0))
            batch_target_renderings.append(torch.cat(target_renderings, dim=0))

        epsilon_render    = 0.1
        batch_input_renderings_logged  = torch.log(torch.stack(batch_input_renderings, dim=0)  + epsilon_render)
        batch_target_renderings_logged = torch.log(torch.stack(batch_target_renderings, dim=0) + epsilon_render)

        loss = nn.functional.l1_loss(batch_input_renderings_logged, batch_target_renderings_logged)

        return loss

class MixedLoss(nn.Module):
    def __init__(self, l1_weight = 0.1):
        super(MixedLoss, self).__init__()

        self.l1_weight      = l1_weight
        self.l1_loss        = SVBRDFL1Loss()
        self.rendering_loss = RenderingLoss()

    def forward(self, input, target):
        return self.l1_weight * self.l1_loss(input, target) + self.rendering_loss(input, target)