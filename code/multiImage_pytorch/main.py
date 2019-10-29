import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import model
import losses

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

writer = SummaryWriter("./logs")

generator = model.Generator(12).cuda()
print(generator)

inputs, targets = read_data(["./in1.png", "./in2.png", "./in3.png"])
criterion = losses.SVBRDFL1Loss()
optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
for epoch in range(100):

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

writer.add_graph(generator, inputs.cuda())
writer.close()

fig=plt.figure(figsize=(8, 8))
for i in range(inputs.shape[0]):
    input = inputs[i].unsqueeze(0).cuda()
    #target = targets[i]

    output = generator(input)
             
    images = torch.cat(output.split(3, dim=1), dim=0).clone().cpu().detach().permute(0, 2, 3, 1)

    fig.add_subplot(inputs.shape[0], 5, i * 5 + 1)
    plt.imshow(gamma_encode(inputs[i].clone().cpu().detach().permute(1, 2, 0)))

    fig.add_subplot(inputs.shape[0], 5, i * 5 + 2)
    plt.imshow(gamma_encode(images[0]))

    fig.add_subplot(inputs.shape[0], 5, i * 5 + 3)
    plt.imshow(gamma_encode(images[1]))

    fig.add_subplot(inputs.shape[0], 5, i * 5 + 4)
    plt.imshow(gamma_encode(images[2]))

    fig.add_subplot(inputs.shape[0], 5, i * 5 + 5)
    plt.imshow(gamma_encode(images[3]))
plt.show()
