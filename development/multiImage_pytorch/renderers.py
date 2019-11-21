import math
import utils
import torch

def dot_product(a, b):
    return torch.sum(torch.mul(a, b), dim=-3, keepdim=True)

def normalize(a):
    return torch.div(a, torch.sqrt(dot_product(a, a)))

class LocalRenderer:
    def xi(self, x):
        return (x > 0.0) * torch.ones_like(x)

    def compute_diffuse_term(self, diffuse, ks):
        kd = (1.0 - ks)
        return  kd * diffuse / math.pi

    def compute_microfacet_distribution(self, roughness, NH):
        alpha            = roughness**2
        alpha_squared    = alpha**2 
        NH_squared       = NH**2
        denominator_part = torch.clamp(NH_squared * (alpha_squared + (1 - NH_squared) / NH_squared), min=0.001)
        return (alpha_squared * self.xi(NH)) / (math.pi * denominator_part**2)

    def compute_fresnel(self, specular, VH):
        return specular + (1.0 - specular) * (1.0 - VH)**5

    def compute_g1(self, roughness, XH, XN):
        alpha         = roughness**2
        alpha_squared = alpha**2
        XN_squared    = XN**2
        return 2 * self.xi(XH / XN) / (1 + torch.sqrt(1 + alpha_squared * (1.0 - XN_squared) / XN_squared))

    def compute_geometry(self, roughness, VH, LH, VN, LN):
        return self.compute_g1(roughness, VH, VN) * self.compute_g1(roughness, LH, LN)

    def compute_specular_term(self, wi, wo, normals, diffuse, roughness, specular):
        # Compute the half direction
        H = normalize((wi + wo) / 2.0)

        # Precompute some dot product
        NH  = dot_product(normals, H)
        VH  = dot_product(wo, H)
        LH  = dot_product(wi, H)
        VN  = dot_product(wo, normals) 
        LN  = dot_product(wi, normals)

        F = self.compute_fresnel(specular, VH)
        G = self.compute_geometry(roughness, VH, LH, VN, LN)
        D = self.compute_microfacet_distribution(roughness, NH)
        
        # We treat the fresnel term as the portion of light that is reflected
        return F * G * D / (4.0 * VN * LN), F

    def evaluate_brdf(self, wi, wo, normals, diffuse, roughness, specular):
        specular_term, ks = self.compute_specular_term(wi, wo, normals, diffuse, roughness, specular)
        diffuse_term      = self.compute_diffuse_term(diffuse, ks)
        return diffuse_term + specular_term

    def render(self, scene, svbrdf):

        # Generate surface coordinates for the material patch
        # The center point of the patch is located at (0, 0, 0) which is the center of the global coordinate system.
        # The patch itself spans from (-1, -1, 0) to (1, 1, 0).
        xcoords_row  = torch.linspace(-1, 1, svbrdf.shape[-1])
        xcoords      = xcoords_row.unsqueeze(0).expand(svbrdf.shape[-2], svbrdf.shape[-1]).unsqueeze(0)
        ycoords      = -1 * torch.transpose(xcoords, dim0=1, dim1=2)
        coords       = torch.cat((xcoords, ycoords, torch.zeros_like(xcoords)), dim=0)

        # [x,y,z] (shape = (3)) -> [[[x]], [[y]], [[z]]] (shape = (3, 1, 1))
        camera_pos          = scene.camera.pos.unsqueeze(-1).unsqueeze(-1)
        # We treat the center of the material patch as focal point of the camera
        relative_camera_pos = camera_pos - coords
        wo                  = normalize(relative_camera_pos)

        normals, diffuse, roughness, specular = utils.unpack_svbrdf(svbrdf)

        # Avoid zero roughness (i. e., potential division by zero)
        roughness = torch.clamp(roughness, min=0.001)

        # For each light do:
        # [x,y,z] (shape = (3)) -> [[[x]], [[y]], [[z]]] (shape = (3, 1, 1))
        light_pos          = scene.light.pos.unsqueeze(-1).unsqueeze(-1)
        relative_light_pos = light_pos - coords
        wi                 = normalize(relative_light_pos)

        f        = self.evaluate_brdf(wi, wo, normals, diffuse, roughness, specular)
        wi_dot_N = torch.clamp(dot_product(wi, normals), min=0.0) # Only consider the upper hemisphere

        light_color = scene.light.color.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        falloff     = 1.0 / torch.sqrt(dot_product(relative_light_pos, relative_light_pos))**2     # Radial light intensity falloff
        radiance    = torch.mul(torch.mul(f, light_color * falloff), wi_dot_N)

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
    scene    = Scene(Camera([0.0, 0.0, 2.0]), Light([-1.0, -1.0, 2.0], [50.0, 50.0, 50.0]))

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