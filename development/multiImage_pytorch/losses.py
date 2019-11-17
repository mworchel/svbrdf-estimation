import torch
import torch.nn as nn
import utils

class SVBRDFL1Loss(nn.Module):
    def forward(self, input, target):
        # Split the SVBRDF into its individual components
        input_normals,  input_diffuse,  input_roughness,  input_specular  = utils.unpack_svbrdf(input)
        target_normals, target_diffuse, target_roughness, target_specular = utils.unpack_svbrdf(target)

        return nn.functional.l1_loss(input_normals, target_normals) + nn.functional.l1_loss(input_diffuse, target_diffuse) + nn.functional.l1_loss(input_roughness, target_roughness) + nn.functional.l1_loss(input_specular, target_specular)
