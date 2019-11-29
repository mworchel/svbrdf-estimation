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

        # Create multiple (test) scenes
        self.scenes   = [
            renderers.Scene(
                renderers.Camera([0.0, 0.0, 2.0]), 
                renderers.Light([-1.0, -1.0, 2.0], [50.0, 50.0, 50.0])),
            renderers.Scene(
                renderers.Camera([1.0, 0.0, 2.0]), 
                renderers.Light([ 1.0,  1.0, 2.0], [50.0, 50.0, 50.0])),
            renderers.Scene(
                renderers.Camera([1.0, 0.0, 2.0]), 
                renderers.Light([ 0.0,  1.0, 2.0], [5.0, 5.0, 5.0]))                
        ]

    def forward(self, input, target):
        # Generate one input and one target rendering for each scene
        input_renderings  = []
        target_renderings = []
        for scene in self.scenes:
            input_renderings.append(self.renderer.render(scene, input).unsqueeze(0))
            target_renderings.append(self.renderer.render(scene, target).unsqueeze(0))

        input_renderings  = torch.cat(input_renderings, dim=0)
        target_renderings = torch.cat(target_renderings, dim=0)

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