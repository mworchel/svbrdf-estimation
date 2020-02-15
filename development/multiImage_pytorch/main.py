import cli
import dataset
import json
import losses
import math
import models
import os
from tensorboardX import SummaryWriter
import torch
import utils

args = cli.parse_args()

# Make the result reproducible
utils.enable_deterministic_random_engine()

# Create the model
model          = models.SingleViewModel().cuda()
training_state = {'epoch' : 0}

# Load the model and training state on demand
model_dir = os.path.abspath(args.model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

training_state_path = os.path.join(model_dir, "state.json")
model_path          = os.path.join(model_dir, "model.data")
if os.path.exists(model_path):
    if not args.retrain:
        model.load_state_dict(torch.load(model_path))
else:
    if args.mode == 'test':
        print("No model found in the model directory but it is required for testing.")
        exit(1)
    else:
        print("No model found in the model directory. Performing retraining.")
        args.retrain = True

if os.path.exists(training_state_path):
    if not args.retrain:
        with open(training_state_path, 'r') as f:
            training_state = json.load(f)
        print("Loaded training state: {:s}.".format(str(training_state)))

# TODO: Choose a random number for the used input image count if we are training and we don't request it to be fix (see fixImageNb for reference)
data = dataset.SvbrdfDataset(data_directory=args.input_dir, image_size=args.image_size, input_image_count=args.image_count, used_input_image_count=args.used_image_count, use_augmentation=True, mix_materials=args.mode=='train')

if args.mode == 'train':
    validation_split = 0.01
    print("Using {:.2f} % of the data for validation".format(round(validation_split * 100.0, 2)))
    training_data, validation_data = torch.utils.data.random_split(data, [int(math.ceil(len(data) * (1.0 - validation_split))), int(math.floor(len(data) * validation_split))])
    print("Training samples: {:d}.".format(len(training_data)))
    print("Validation samples: {:d}.".format(len(validation_data)))

    if len(validation_data) == 0:
        # Fixed fallback if the training set is too small
        print("Training dataset too small for validation split. Using training data for validation.")
        validation_data = training_data

    training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=8, pin_memory=True, shuffle=True)
    batch_count         = int(math.ceil(len(training_data) / training_dataloader.batch_size))

    # Determine the epoch range
    epoch_start = training_state['epoch']
    epoch_end   = args.epochs

    print("Training from epoch {:d} to {:d}".format(epoch_start, epoch_end))

    # Set up the optimizer and loss
    optimizer     = torch.optim.Adam(model.parameters(), lr=1e-5)
    # TODO: Use scheduler if necessary
    #scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min') 
    loss_function = losses.MixedLoss()

    # Setup statistics stuff
    statistics_dir    = os.path.abspath("./logs")
    if not os.path.exists(statistics_dir):
        os.makedirs(statistics_dir)
    writer            = SummaryWriter(statistics_dir)
    last_batch_inputs = None

    model.train(True)
    for epoch in range(epoch_start, epoch_end):
        for i, batch in enumerate(training_dataloader):
            # Unique index of this batch
            batch_index = epoch * batch_count + i

            # Construct inputs
            batch_inputs = batch["inputs"].cuda()
            batch_svbrdf = batch["svbrdf"].cuda()

            # Perform a step
            optimizer.zero_grad() 
            outputs = model(batch_inputs)
            loss    = loss_function(outputs, batch_svbrdf)
            loss.backward()
            optimizer.step()

            print("Epoch {:d}, Batch {:d}, loss: {:f}".format(epoch + 1, i + 1, loss.item()))

            # Statistics
            writer.add_scalar("loss", loss.item(), batch_index)
            last_batch_inputs = batch_inputs

        if epoch % args.save_frequency == 0:
            torch.save(model.state_dict(), model_path)

            training_state['epoch'] = epoch
            with open(training_state_path, 'w') as f:
                json.dump(training_state, f)

    model.train(False)

    # Save a final snapshot of the model
    torch.save(model.state_dict(), model_path)
    training_state['epoch'] = epoch_end
    with open(training_state_path, 'w') as f:
        json.dump(training_state, f)

    # FIXME: This does not work with the last conv layers on both the single-view and multi-view model
    #writer.add_graph(model, last_batch_inputs) 
    writer.close()

    # Use the validation dataset as test data
    test_data = validation_data 
else:
    test_data = data

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, pin_memory=True)

# Plotting
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(8, 8))
row_count = 2 * len(test_data)
col_count = 5
for i_row, batch in enumerate(test_dataloader):
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
