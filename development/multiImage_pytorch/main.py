import dataset
import losses
import matplotlib.pyplot as plt
import models
import numpy as np
import os
import random
import renderers
from tensorboardX import SummaryWriter
import torch
import utils

# Make the result reproducible
seed = 313
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

# Create the model
model = models.SingleViewModel().cuda() 
print(model)

train_data       = dataset.SvbrdfDataset(data_directory="./data/train", input_image_count=10, used_input_image_count=1)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, pin_memory=True)

# Load on demand
load_model = True
model_dir  = "./models"
model_path = os.path.join(model_dir, "model.model")
criterion = losses.MixedLoss()
if load_model:
    model.load_state_dict(torch.load(model_path))
else:
    model.train(True)
    writer = SummaryWriter("./logs")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    last_batch_inputs = None
    for epoch in range(3000):
        for batch in train_dataloader:
            # Construct inputs
            batch_inputs = batch["inputs"].cuda()
            batch_svbrdf = batch["svbrdf"].cuda()

            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_svbrdf)
            loss.backward()
            optimizer.step()    # Does the update

            writer.add_scalar("loss", loss.item(), epoch)

            print("Epoch {:d}, loss: {:f}".format(epoch + 1, loss.item()))

            last_batch_inputs = batch_inputs
    model.train(False)

    # Save a snapshot of the model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_path)
    #writer.add_graph(model, last_batch_inputs)
    writer.close()

test_data      = dataset.SvbrdfDataset(data_directory="./data/test", input_image_count=10, used_input_image_count=1)
all_dataloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([train_data, test_data]), batch_size=1, pin_memory=True)

fig=plt.figure(figsize=(8, 8))
row_count = 2 * (len(train_data) + len(test_data))
col_count = 5
for i_row, batch in enumerate(all_dataloader):
    # Construct inputs
    batch_inputs = batch["inputs"].cuda()
    batch_svbrdf = batch["svbrdf"].cuda()

    outputs = model(batch_inputs)

    # for i, phi in enumerate(np.linspace(0.0, np.pi*2, 120, endpoint=True)):
    #     c = renderers.Camera([1.7 * np.cos(phi), 1.7 * np.sin(phi), 1.7])
    #     l = renderers.Light([2.0, 0.0, 2.0], [30, 30, 30])
    #     s = renderers.Scene(c, l)
    #     r = utils.gamma_encode(criterion.rendering_loss.renderer.render(s, outputs).cpu().detach().squeeze(0).permute(1, 2, 0))
    #     m = renderers.OrthoToPerspectiveMapping(c, (640, 480))
    #     r = m.apply(r.numpy())

    #     r_dir = "rendering_l1_{:d}".format(i_row)
    #     if not os.path.exists(r_dir):
    #         os.makedirs(r_dir)
    #     plt.imsave(os.path.join(r_dir, "{:d}.png".format(i)), r)

    rendered1 = utils.gamma_encode(criterion.rendering_loss.renderer.render(criterion.rendering_loss.scenes[0], outputs).cpu().detach().squeeze(0).permute(1, 2, 0))
    m1 = renderers.OrthoToPerspectiveMapping(criterion.rendering_loss.scenes[0].camera, (500, 500))
    rendered1 = m1.apply(rendered1.numpy())

    rendered2 = utils.gamma_encode(criterion.rendering_loss.renderer.render(criterion.rendering_loss.scenes[1], outputs).cpu().detach().squeeze(0).permute(1, 2, 0))
    m2 = renderers.OrthoToPerspectiveMapping(criterion.rendering_loss.scenes[1].camera, (500, 500))
    rendered2 = m2.apply(rendered2.numpy())

    rendered3 = utils.gamma_encode(criterion.rendering_loss.renderer.render(criterion.rendering_loss.scenes[2], outputs).cpu().detach().squeeze(0).permute(1, 2, 0))
    m3 = renderers.OrthoToPerspectiveMapping(criterion.rendering_loss.scenes[2].camera, (500, 500))
    rendered3 = m3.apply(rendered3.numpy())

    plt.imsave("{:d}_r1.png".format(i_row), rendered1)
    plt.imsave("{:d}_r2.png".format(i_row), rendered2)
    plt.imsave("{:d}_r3.png".format(i_row), rendered3)

    input       = utils.gamma_encode(batch_inputs.squeeze(0)[0]).cpu().permute(1, 2, 0)
    reeeendered = criterion.rendering_loss.renderer.render(criterion.rendering_loss.scenes[0], outputs).cpu().detach().squeeze(0).permute(1, 2, 0)
    target_maps = torch.cat(batch_svbrdf.split(3, dim=1), dim=0).clone().cpu().detach().permute(0, 2, 3, 1)
    output_maps = torch.cat(outputs.split(3, dim=1), dim=0).clone().cpu().detach().permute(0, 2, 3, 1)

    # tmp_target_maps    = target_maps.clone()
    # tmp_target_maps[0] = utils.encode_as_unit_interval(tmp_target_maps[0])
    # tmp_target_maps[1] = utils.gamma_encode(tmp_target_maps[1])
    # tmp_target_maps[3] = utils.gamma_encode(tmp_target_maps[3])
    # tmp_target_svbrdf = torch.cat(torch.chunk(tmp_target_maps, 4, dim=0), -2).squeeze(0)
    # plt.imsave("{:d}_target_svbrdf.png".format(i_row), tmp_target_svbrdf)
    # plt.imsave("{:d}_target_n.png".format(i_row), tmp_target_maps[0])
    # plt.imsave("{:d}_target_d.png".format(i_row), tmp_target_maps[1])
    # plt.imsave("{:d}_target_r.png".format(i_row), tmp_target_maps[2])
    # plt.imsave("{:d}_target_s.png".format(i_row), tmp_target_maps[3])

    # tmp_output_maps    = output_maps.clone()
    # tmp_output_maps[0] = utils.encode_as_unit_interval(tmp_output_maps[0])
    # tmp_output_maps[1] = utils.gamma_encode(tmp_output_maps[1])
    # tmp_output_maps[3] = utils.gamma_encode(tmp_output_maps[3])
    # tmp_output_svbrdf = torch.cat(torch.chunk(tmp_output_maps, 4, dim=0), -2).squeeze(0)
    # plt.imsave("{:d}_output_svbrdf.png".format(i_row), tmp_output_svbrdf)
    # plt.imsave("{:d}_output_n.png".format(i_row), tmp_output_maps[0])
    # plt.imsave("{:d}_output_d.png".format(i_row), tmp_output_maps[1])
    # plt.imsave("{:d}_output_r.png".format(i_row), tmp_output_maps[2])
    # plt.imsave("{:d}_output_s.png".format(i_row), tmp_output_maps[3])

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 1)
    plt.imshow(input)
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 2)
    plt.imshow(utils.encode_as_unit_interval(target_maps[0]))
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 3)
    plt.imshow(target_maps[1])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 4)
    plt.imshow(target_maps[2])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 5)
    plt.imshow(target_maps[3])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 6)
    plt.imshow(utils.gamma_encode(reeeendered))
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 7)
    plt.imshow(utils.encode_as_unit_interval(output_maps[0]))
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 8)
    plt.imshow(output_maps[1])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 9)
    plt.imshow(output_maps[2])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 10)
    plt.imshow(output_maps[3])
    plt.axis('off')
plt.show()
