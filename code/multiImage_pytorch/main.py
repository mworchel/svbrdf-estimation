import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import model
import losses
import os

def gamma_decode(images):
    return torch.pow(images, 2.2)

def gamma_encode(images):
    return torch.pow(images, 1.0/2.2)

def read_data(paths):
    inputs  = None
    targets = None
    for path in paths:
        img  = gamma_decode(torch.Tensor(plt.imread(path)).permute(2, 0, 1))
        imgs = torch.cat(img.unsqueeze(0).split(256, dim=-1), 0)
        input  = imgs[-5].unsqueeze(0)
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


inputs, targets = read_data(["./in1.png", "./in2.png", "./in3.png"])

# Load on demand
load_model = False
model_dir  = "./models"
model_path = os.path.join(model_dir, "generator.model")
if load_model:
    generator.load_state_dict(torch.load(model_path))
else:
    writer = SummaryWriter("./logs")
    criterion = losses.SVBRDFL1Loss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    for epoch in range(500):

        batch_inputs  = inputs[:2].cuda()
        batch_targets = targets[:2].cuda()

        # in your training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        outputs = generator(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()    # Does the update

        writer.add_scalar("loss", loss.item(), epoch)

        print("Epoch {:d}, loss: {:f}".format(epoch + 1, loss.item()))

    # Save a snapshot of the model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(generator.state_dict(), model_path)
    writer.add_graph(generator, inputs.cuda())
    writer.close()

fig=plt.figure(figsize=(8, 8))
row_count = 2 * inputs.shape[0]
col_count = 5
for i_row, i in enumerate(range(inputs.shape[0])):
    output = generator(inputs[i].unsqueeze(0).cuda())

    output_svbrdf = gamma_encode(torch.cat(output.split(3, dim=1), dim=0).clone().cpu().detach().permute(0, 2, 3, 1))
    target_svbrdf = gamma_encode(torch.cat(targets[i].unsqueeze(0).split(3, dim=1), dim=0).clone().permute(0, 2, 3, 1))
    input         = gamma_encode(inputs[i]).permute(1, 2, 0)

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
