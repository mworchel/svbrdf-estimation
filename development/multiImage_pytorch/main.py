import dataset
import losses
import matplotlib.pyplot as plt
import models
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

# Create the model
model = models.SingleViewModel().cuda() 
print(model)

train_data       = dataset.SvbrdfDataset(data_directory="./data/train", input_image_count=10, used_input_image_count=1)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, pin_memory=True)

# Load on demand
load_model = False
model_dir  = "./models"
model_path = os.path.join(model_dir, "model.model")
if load_model:
    model.load_state_dict(torch.load(model_path))
else:
    model.train(True)
    writer = SummaryWriter("./logs")
    criterion = losses.MixedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    last_batch_inputs = None
    for epoch in range(100):
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

    input       = utils.gamma_encode(batch_inputs.squeeze(0)[0]).cpu().permute(1, 2, 0)
    target_maps = torch.cat(batch_svbrdf.split(3, dim=1), dim=0).clone().cpu().detach().permute(0, 2, 3, 1)
    output_maps = torch.cat(outputs.split(3, dim=1), dim=0).clone().cpu().detach().permute(0, 2, 3, 1)

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
