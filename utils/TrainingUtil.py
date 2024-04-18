from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import torch
import time
from tqdm import tqdm


def train_model(model, optimizer, data_train, data_val, num_epochs, batch_size, device='cpu', verbose=True, save=True):
    quiet = not verbose

    # training loop
    model.train()
    loss_tracker = []
    val_loss_tracker = []
    best_epoch, best_loss, best_model = None, None, None
    
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    
    for i in tqdm(range(1, num_epochs + 1), disable=verbose, desc='Training Model', file=sys.stdout):

        if verbose:
            print("-" * 60)
            print("Running Epoch: %02d / %02d" % (i, num_epochs))

        # evaluate per batch
        loss = 0
        for data in tqdm(dataloader, disable=quiet, file=sys.stdout):
            # Send data to CUDA device
            data_device = []
            for i, item in enumerate(data):
                # Segmentation masks are not stored as Tensors because they are all different shapes
                if isinstance(item, torch.Tensor):
                    item = item.to(device)
                data_device.append(item)
            data = data_device
            # forward
            epoch_loss = model(*data)

            # backward
            optimizer.zero_grad()
            epoch_loss.backward()
            optimizer.step()

            loss += epoch_loss.item()

        # compute validation loss
        model.eval()
        with torch.no_grad():
            val_loss = model(data_val.images.to(device), data_val.labels.to(device), data_val.bboxes.to(device))
            val_loss_item = val_loss.item() / data_val.n_samples
        model.train()

        # track loss
        loss /= data_train.n_samples
        loss_tracker.append(loss)
        val_loss_tracker.append(val_loss_item)
        if verbose:
            print("  Training Loss: %.2f, Validation Loss %.2f" % (loss, val_loss_item))

        # save the best model TODO: implement validation loss for this criteria
        if (best_loss is None) or (loss < best_loss):
            best_epoch = i
            best_loss = loss
            best_model = deepcopy(model.state_dict())

    if save:
        # make a save directory
        save_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base_dir = os.path.join("results/models", save_timestamp)
        Path(base_dir).mkdir(parents=True, exist_ok=True)

        # save the model
        model_filename = os.path.join(base_dir, "model.pt")
        torch.save(model.state_dict(), model_filename)
        best_model_filename = os.path.join(base_dir, "model_epoch_{}.pt".format(best_epoch))
        torch.save(best_model, best_model_filename)

        # save the properties
        save_properties(model, optimizer, base_dir)

        # save the loss curve
        save_loss_curve(loss_tracker, val_loss_tracker, base_dir, save_timestamp)

        print("Model, properties, and results saved to: {}".format(base_dir))
    return loss_tracker


def save_properties(model, optimizer, base_dir):

    props_filename = os.path.join(base_dir, "training.properties")
    props_file = open(props_filename, "w")

    # write the model properties
    for param in model.hyper_params.keys():
        props_file.write("{}={}\n".format("model.{}".format(param), model.hyper_params[param]))

    # write the optimizer properties
    for param in ['lr', 'momentum']:
        props_file.write("{}={}\n".format("optim.{}".format(param), optimizer.param_groups[0][param]))

    props_file.close()


def save_loss_curve(loss_tracker, val_loss_tracker, base_dir, save_timestamp):
    loss_filename = os.path.join(base_dir, "loss_{}.png".format(save_timestamp))

    plt.cla()
    plt.plot(loss_tracker, label='Training')
    plt.plot(val_loss_tracker, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(loss_filename)
    plt.cla()
