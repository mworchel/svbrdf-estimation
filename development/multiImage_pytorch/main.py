import dataset
import losses
import matplotlib.pyplot as plt
import model
import numpy as np
import os
import random
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

def read_data(paths):
    inputs  = None
    targets = None
    for path in paths:
        img  = torch.Tensor(plt.imread(path)).permute(2, 0, 1)
        imgs = torch.cat(img.unsqueeze(0).split(256, dim=-1), 0)
        input  = utils.gamma_decode(imgs[-5].unsqueeze(0))
        target = torch.cat(imgs[10:].split(1, dim=0), dim=1)
        
        if inputs is None:
            inputs = input
        else:
            inputs = torch.cat((inputs, input), dim=0)

        if targets is None:
            targets = target
        else:
            targets = torch.cat((targets, target), dim=0)

    return inputs, targets


generator = model.Generator(12).cuda()
print(generator)

train_data = dataset.SvbrdfDataset(data_directory="./data/train", input_image_count=10, used_input_image_count=1)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, pin_memory=True)

inputs, targets = read_data(["./in1.png", "./in2.png", "./in3.png"])

# Load on demand
load_model = False
model_dir  = "./models"
model_path = os.path.join(model_dir, "generator.model")
if load_model:
    generator.load_state_dict(torch.load(model_path))
else:
    generator.train(True)
    writer = SummaryWriter("./logs")
    criterion = losses.SVBRDFL1Loss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    for epoch in range(1):
        for batch in train_dataloader:
            # Construct inputs
            batch_inputs = batch["inputs"].cuda()
            batch_svbrdf = batch["svbrdf"].cuda()

            # We know we only require one input image
            batch_inputs.squeeze_(1)

            # batch_inputs  = inputs[:2].cuda()
            # batch_targets = targets[:2].cuda()

            # in your training loop:
            optimizer.zero_grad()   # zero the gradient buffers
            outputs = generator(batch_inputs)
            loss = criterion(outputs, batch_svbrdf)
            loss.backward()
            optimizer.step()    # Does the update

            writer.add_scalar("loss", loss.item(), epoch)

            print("Epoch {:d}, loss: {:f}".format(epoch + 1, loss.item()))
    generator.train(False)

    # Save a snapshot of the model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(generator.state_dict(), model_path)
    writer.add_graph(generator, inputs.cuda())
    writer.close()

test_data = dataset.SvbrdfDataset(data_directory="./data/test", input_image_count=10, used_input_image_count=1)
test_dataloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([train_data, test_data]), batch_size=1, pin_memory=True)

fig=plt.figure(figsize=(8, 8))
row_count = 2 * (len(train_data) + len(test_data))# inputs.shape[0]
col_count = 5

for i_row, batch in enumerate(test_dataloader):
    # Construct inputs
    batch_inputs = batch["inputs"].cuda()
    batch_svbrdf = batch["svbrdf"].cuda()

    # We know we only have one input image
    batch_inputs.squeeze_(1)

    output = generator(batch_inputs).squeeze(0)

for i_row, i in enumerate(range(inputs.shape[0])):
    output = generator(inputs[i].unsqueeze(0).cuda())

    output_svbrdf = torch.cat(output.split(3, dim=1), dim=0).clone().cpu().detach().permute(0, 2, 3, 1)
    target_svbrdf = torch.cat(targets[i].unsqueeze(0).split(3, dim=1), dim=0).clone().permute(0, 2, 3, 1)
    input         = utils.gamma_encode(inputs[i]).permute(1, 2, 0)

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 1)
    plt.imshow(input)
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 2)
    plt.imshow(target_svbrdf[0])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 3)
    plt.imshow(target_svbrdf[1])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 4)
    plt.imshow(target_svbrdf[2])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 5)
    plt.imshow(target_svbrdf[3])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 7)
    plt.imshow(output_svbrdf[0])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 8)
    plt.imshow(output_svbrdf[1])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 9)
    plt.imshow(output_svbrdf[2])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 10)
    plt.imshow(output_svbrdf[3])
    plt.axis('off')
plt.show()
