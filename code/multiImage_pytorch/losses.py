import torch
import torch.nn as nn

class SVBRDFL1Loss(nn.Module):
    def forward(self, input, target):
        # Split the SVBRDF into its individual components
        input_normals   = self.get_normals(input)
        input_diffuse   = self.get_diffuse_albedo(input)
        input_roughness = self.get_roughness(input)
        input_specular  = self.get_specular_albedo(input)

        target_normals   = self.get_normals(target)
        target_diffuse   = self.get_diffuse_albedo(target)
        target_roughness = self.get_roughness(target)
        target_specular  = self.get_specular_albedo(target)

        return nn.functional.l1_loss(input_normals, target_normals) + nn.functional.l1_loss(input_diffuse, target_diffuse) + nn.functional.l1_loss(input_roughness, target_roughness) + nn.functional.l1_loss(input_specular, target_specular)

    def get_normals(self, input):
        return input[:,0:3,:,:]

    def get_diffuse_albedo(self, input):
        return input[:,3:6,:,:]

    def get_specular_albedo(self, input):
        return input[:,9:12,:,:]

    def get_roughness(self, input):
        return input[:,6:9,:,:]
