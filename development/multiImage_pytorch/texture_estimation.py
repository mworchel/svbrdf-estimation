import pyredner
import matplotlib.pyplot as plt
import torch
import environment as env
from renderers import RednerRenderer, LocalRenderer
import pathlib
import utils

input_image_count = 10

# Load a SVBRDF
file_path    = R"./data/train/10_1_parquet_floor_0.png"
full_image   = torch.Tensor(plt.imread(file_path)).permute(2, 0, 1)

# Split the full image apart along the horizontal direction 
# Magick number 4 is the number of maps in the SVBRDF
image_parts  = torch.cat(full_image.unsqueeze(0).chunk(input_image_count + 4, dim=-1), 0) # [n, 3, 256, 256]

# Query the height of the images, which represents the size of the read images
# (assuming square images)
actual_image_size = image_parts.shape[-2]

# Determine the top left point of the cropped image
# TODO: If we use jittering, this has to be determined by the random jitter of the rendered images
crop_anchor = torch.IntTensor([0, 0])

# Read and crop the SVBRDF
normals   = image_parts[input_image_count + 0]
normals   = utils.decode_from_unit_interval(normals)
diffuse   = image_parts[input_image_count + 1]
roughness = image_parts[input_image_count + 2]
specular  = image_parts[input_image_count + 3]

target_svbrdf = utils.pack_svbrdf(normals, diffuse, roughness, specular) # [12, 256, 256]

# Target rendering
camera = env.Camera([0, 0, 2])
light  = env.Light([1, 1, 0.5], [20, 20, 20])
scene  = env.Scene(camera, light)
renderer = RednerRenderer()
target_rendering = renderer.render(scene, target_svbrdf).squeeze(0)

def save_image(path, tensor):
    plt.imsave(path, torch.clamp(utils.gamma_encode(tensor.permute(1, 2, 0)), min=0.0, max=1.0).numpy())

def save_svbrdf(path, svbrdf):
    normals, diffuse, roughness, specular = utils.unpack_svbrdf(svbrdf)
    normals    = utils.encode_as_unit_interval(normals)
    svbrdf_row = torch.cat([normals, diffuse, roughness, specular], dim=-1)
    plt.imsave(path, torch.clamp(svbrdf_row.permute(1, 2, 0), min=0.0, max=1.0).numpy())

output_dir = pathlib.Path("./tmp2")
output_dir.mkdir(parents=True, exist_ok=True)
save_svbrdf(str(output_dir.joinpath("target_svbrdf.png")), target_svbrdf)
save_image(str(output_dir.joinpath("target.png")), target_rendering)

def decode_normals(normals_xy):
    normals_x, normals_y = torch.split(normals_xy.mul(3.0), 1, dim=-3)
    normals_z            = torch.ones_like(normals_x)
    normals              = torch.cat([normals_x, normals_y, normals_z], dim=-3)
    norm                 = torch.sqrt(torch.sum(torch.pow(normals, 2.0), dim=-3, keepdim=True))
    return torch.div(normals, norm)

# Stuff to optimize
estimated_encoded_normals = torch.rand((2, normals.shape[1], normals.shape[2]), requires_grad=True)
#estimated_normals = torch.rand_like(normals, requires_grad=True)
#estimated_diffuse = torch.rand_like(diffuse, requires_grad=True)
#estimated_roughness = torch.rand_like(roughness, requires_grad=True)
optimizer = torch.optim.Adam([estimated_encoded_normals], 1e-2)

for i in range(80):
    optimizer.zero_grad()
    svbrdf    = utils.pack_svbrdf(decode_normals(estimated_encoded_normals), diffuse, roughness, specular)
    rendering = renderer.render(scene, svbrdf).squeeze(0)
    save_image(str(output_dir.joinpath("step_{:d}.png".format(i))), rendering.detach())
    loss      = torch.nn.functional.l1_loss(target_rendering, rendering)
    loss.backward()
    optimizer.step()
    save_svbrdf(str(output_dir.joinpath("svbrdf_{:d}.png".format(i))), svbrdf.detach())
    print("Loss ({:d}): {:f}".format(i, loss.item()))
    