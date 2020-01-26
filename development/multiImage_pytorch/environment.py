import torch
import utils

class Camera:
    def __init__(self, pos):
        self.pos = pos

class Light:
    def __init__(self, pos, color):
        self.pos   = pos
        self.color = color

class Scene:
    def __init__(self, camera, light):
        self.camera = camera
        self.light  = light

def generate_random_scenes(count):
    # Randomly distribute both, view and light positions
    view_positions  = utils.generate_normalized_random_direction(count, 0.001, 0.1) # shape = [count, 3]
    light_positions = utils.generate_normalized_random_direction(count, 0.001, 0.1)

    scenes = []
    for i in range(count):
        c = Camera(view_positions[i])
        # Light has lower power as the distance to the material plane is not as large
        l = Light(light_positions[i], [20.0, 20.0, 20.0]) 
        scenes.append(Scene(c, l))

    return scenes

def generate_specular_scenes(count):
    # Only randomly distribute view positions and place lights in a perfect mirror configuration
    view_positions  = utils.generate_normalized_random_direction(count, 0.001, 0.1) # shape = [count, 3]
    light_positions = view_positions * torch.Tensor([-1.0, -1.0, 1.0]).unsqueeze(0)

    # Reference: "parameters chosen empirically to have a nice distance from a -1;1 surface.""
    distance_view  = torch.exp(torch.Tensor(count, 1).normal_(mean=0.5, std=0.75)) 
    distance_light = torch.exp(torch.Tensor(count, 1).normal_(mean=0.5, std=0.75))

    # Reference: "Shift position to have highlight elsewhere than in the center."
    shift = torch.cat([torch.Tensor(count, 2).uniform_(-1.0, 1.0), torch.zeros((count, 1)) + 0.0001], dim=-1)

    view_positions  = view_positions  * distance_view  + shift
    light_positions = light_positions * distance_light + shift

    scenes = []
    for i in range(count):
        c = Camera(view_positions[i])
        l = Light(light_positions[i], [50.0, 50.0, 50.0])
        scenes.append(Scene(c, l))

    return scenes

if __name__ == '__main__':
    import dataset
    import os
    import matplotlib.pyplot as plt
    import renderers

    data   = dataset.SvbrdfDataset(data_directory="./data/train", image_size=256, input_image_count=10, used_input_image_count=1, use_augmentation=True)
    loader = torch.utils.data.DataLoader(data, batch_size=1, pin_memory=False)

    renderer = renderers.LocalRenderer()

    # Generate some random scenes
    scenes   = generate_random_scenes(5) + generate_specular_scenes(5)

    def render_scenes(scenes, tag):
        for i, scene in enumerate(scenes):
            # Get one sample
            sample = data[1]
            inputs = sample["inputs"]
            svbrdf = sample["svbrdf"]

            rendering = renderer.render(scene, svbrdf).squeeze(0)
            rendering = utils.gamma_encode(rendering.permute(1, 2, 0))

            # Define the perspective mapping
            perspective_mapping = renderers.OrthoToPerspectiveMapping(scene.camera, (600, 600))

            output_dir = "./scene_tests/{:s}".format(tag)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            plt.imsave(os.path.join(output_dir, "{:d}_ortho_rendering.png".format(i)), rendering)
            plt.imsave(os.path.join(output_dir, "{:d}_persp_rendering.png".format(i)), perspective_mapping.apply(rendering.numpy()))

    render_scenes(generate_random_scenes(5),   "random")
    render_scenes(generate_specular_scenes(5), "specular")