import math
import utils
import torch

def dot_product(a, b):
    return torch.sum(torch.mul(a, b), dim=-3, keepdim=True)

def normalize(a):
    return torch.div(a, torch.sqrt(dot_product(a, a)))

class LocalRenderer:
    def compute_diffuse_term(self, diffuse, specular):
        return diffuse * (1.0 - specular) / math.pi

    def compute_microfacet_distribution(self, roughness, NH):
        alpha         = roughness**2
        alpha_squared = alpha**2
        NH_squared    = NH**2
        denominator   = NH_squared * alpha_squared + (1.0 - NH_squared)
        return alpha_squared / (math.pi * denominator * denominator)

    def compute_specular_term(self, wi, wo, normals, diffuse, roughness, specular):
        # Compute the half direction
        H = normalize((wi + wo) / 2.0)

        # Precompute some dot product
        NH  = dot_product(normals, H)
        VH  = dot_product(wo, H)
        NL  = dot_product(normals, wi)
        NV  = dot_product(normals, wo) 

        D = self.compute_microfacet_distribution(roughness, NH)

        # Cook-Torrance model
        #return F * G * D * 0.25
        
        # Temporarily use Blinn-Phong model
        return specular * torch.pow(NH, (1.0 - roughness) * 125) 

    def evaluate_brdf(self, wi, wo, normals, diffuse, roughness, specular):
        diffuse_term  = self.compute_diffuse_term(diffuse, specular)
        specular_term = self.compute_specular_term(wi, wo, normals, diffuse, roughness, specular)
        return diffuse_term + specular_term

    def render(self, scene, svbrdf):
        # [x,y,z] (shape = (3)) -> [[[x]], [[y]], [[z]]] (shape = (3, 1, 1))
        camera = scene.camera.pos.unsqueeze(-1).unsqueeze(-1)

        # Generate surface coordinates for the material patch
        # The center point of the patch is located at (0, 0, 0) which is the center of the global coordinate system.
        # The patch itself spans from (-1, -1, 0) to (1, 1, 0).
        xcoords_row  = torch.linspace(-1, 1, svbrdf.shape[-1])
        xcoords      = xcoords_row.unsqueeze(0).expand(svbrdf.shape[-2], svbrdf.shape[-1]).unsqueeze(0)
        ycoords      = -1 * torch.transpose(xcoords, dim0=1, dim1=2)
        coords       = torch.cat((xcoords, ycoords, torch.zeros_like(xcoords)), dim=0)

        # We treat the center of the material patch as focal point of the camera
        wo = normalize(camera - coords)

        normals, diffuse, roughness, specular = utils.unpack_svbrdf(svbrdf)

        # For each light do:
        light  = scene.light.pos.unsqueeze(-1).unsqueeze(-1)
        wi     = normalize(light - coords)

        f        = self.evaluate_brdf(wi, wo, normals, diffuse, roughness, specular)
        wi_dot_N = torch.clamp(dot_product(wi, normals), min=0.0) # Only consider the upper hemisphere

        light_color = scene.light.color.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        radiance = torch.mul(torch.mul(f, light_color), wi_dot_N)

        return radiance

class Camera:
    def __init__(self, pos):
        self.pos = torch.Tensor(pos)

class Light:
    def __init__(self, pos, color):
        self.pos   = torch.Tensor(pos)
        self.color = torch.Tensor(color)

class Scene:
    def __init__(self, camera, light):
        self.camera = camera
        self.light  = light

if __name__ == '__main__':
    # Testing code for the renderer(s)
    import dataset
    import matplotlib.pyplot as plt
    import utils

    data   = dataset.SvbrdfDataset(data_directory="./data/train", input_image_count=10, used_input_image_count=1)
    loader = torch.utils.data.DataLoader(data, batch_size=1, pin_memory=False)

    renderer = LocalRenderer()
    scene    = Scene(Camera([0.0, 0.0, 2.0]), Light([0.0, 0.0, 2.0], [5.0, 5.0, 5.0]))

    fig = plt.figure(figsize=(8, 8))
    row_count = len(data)
    col_count = 6
    for i_row, batch in enumerate(loader):
        batch_inputs = batch["inputs"]
        batch_svbrdf = batch["svbrdf"]

        # We only have one image in the inputs
        batch_inputs.squeeze_(0)

        input       = utils.gamma_encode(batch_inputs)
        svbrdf      = batch_svbrdf

        normals_packed, diffuse, roughness, specular = utils.unpack_svbrdf(svbrdf)
        normals   = normals_packed * 2.0 - 1.0
        rendering = utils.gamma_encode(renderer.render(scene, utils.pack_svbrdf(normals, diffuse, roughness, specular)))

        fig.add_subplot(row_count, col_count, i_row * col_count + 1)
        plt.imshow(rendering.squeeze(0).permute(1, 2, 0))
        plt.axis('off')

        fig.add_subplot(row_count, col_count, i_row * col_count + 2)
        plt.imshow(input.squeeze(0).permute(1, 2, 0))
        plt.axis('off')

        fig.add_subplot(row_count, col_count, i_row * col_count + 3)
        plt.imshow(normals_packed.squeeze(0).permute(1, 2, 0))
        plt.axis('off')

        fig.add_subplot(row_count, col_count, i_row * col_count + 4)
        plt.imshow(diffuse.squeeze(0).permute(1, 2, 0))
        plt.axis('off')

        fig.add_subplot(row_count, col_count, i_row * col_count + 5)
        plt.imshow(roughness.squeeze(0).permute(1, 2, 0))
        plt.axis('off')

        fig.add_subplot(row_count, col_count, i_row * col_count + 6)
        plt.imshow(specular.squeeze(0).permute(1, 2, 0))
        plt.axis('off')
    plt.show()