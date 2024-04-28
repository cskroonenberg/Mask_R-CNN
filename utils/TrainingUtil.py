from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
from networks.FasterRCNN import FasterRCNN
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
import torch
import time
from tqdm import tqdm
from utils import EvalUtil, MaskPlot

def train_model(model, optimizer, scheduler, data_train, data_val, num_epochs, batch_size, id2str, device='cpu', verbose=True, save=True, mask_plot_freq=20):
    quiet = not verbose

    is_fcn = isinstance(model, FasterRCNN)
    if is_fcn:
        loss_keys = ['rpn_class', 'rpn_box', 'cls_class', 'cls_box']
    else:
        loss_keys = ['rpn_loss', 'class_loss', 'mask_loss']

    if save:
        # make a save directory
        save_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base_dir = os.path.join("results/models", save_timestamp)
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_dir = os.path.join(base_dir, "checkpoints")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        loss_dir = os.path.join(base_dir, "loss")
        Path(loss_dir).mkdir(parents=True, exist_ok=True)
        viz_dir = os.path.join(base_dir, "viz")
        Path(viz_dir).mkdir(parents=True, exist_ok=True)

    # training loop
    model.train()
    loss_tracker = []
    val_loss_tracker = []
    mAP_avg_tracker = []
    mAP_per_tracker = []
    best_epoch, best_mAP, best_model = None, None, None
    
    train_losses_tracker = {loss_key: [] for loss_key in loss_keys}
    val_losses_tracker = {loss_key: [] for loss_key in loss_keys}
    
    print(f"len(data_train): {len(data_train)}")
    
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Use training data for validation
    # dataloader_val = torch.utils.data.DataLoader(deepcopy(data_train), batch_size=batch_size, shuffle=False, drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False, drop_last=True)
    
    for i in tqdm(range(1, num_epochs + 1), disable=verbose, desc='Training Model', file=sys.stdout):
        model.train()
        if verbose:
            print("-" * 60)
            print("Running Epoch: %02d / %02d" % (i, num_epochs))

        # evaluate per batch
        loss = 0
        train_losses = {loss_key: 0 for loss_key in loss_keys}
    
        loss_type = "all loss"
    
        for data in tqdm(dataloader, disable=quiet, desc='Training Model', file=sys.stdout):
            # # Send data to CUDA device           
            data = [x.to(device, non_blocking=True) for x in data]
            
            try:
                # forward
                if is_fcn:
                    data.append(True)
                    epoch_loss, epoch_losses = model(*data)
                    for loss_type in train_losses.keys():
                        train_losses[loss_type] += epoch_losses[loss_type]
                else:
                    rpn_loss, class_loss, mask_loss = model(*data)
                    epoch_loss = rpn_loss + class_loss + mask_loss

                    # if i < 2 * ((num_epochs + 1) / 3):
                    #     if i % 2 == 0:
                    #         epoch_loss = class_loss
                    #         loss_type = "class_loss"
                    #     else:
                    #         epoch_loss = rpn_loss
                    #         loss_type = "rpn_loss"
                    # else:
                    #     model.rpn.requires_grad = False
                    #     model.classifier.requires_grad = False
                    #     # model.mask_head.requires_grad = True
                    #     epoch_loss = mask_loss   
                    #     loss_type = "rpmask_lossn_loss"
                        
                    train_losses['rpn_loss'] += rpn_loss.item()
                    train_losses['class_loss'] += class_loss.item()
                    train_losses['mask_loss'] += mask_loss.item()
                    
                # backward
                optimizer.zero_grad()
                epoch_loss.backward()
                optimizer.step()

                loss += epoch_loss.item()
            except ValueError as e:
                # print(e)
                pass

        print(f"{loss_type} backprop")

        # compute validation loss
        model.eval()
        val_loss = 0

        batch_truth_boxes = []
        batch_truth_labels = []
        batch_eval_boxes = []
        batch_eval_labels = []
        if is_fcn:
            val_losses = {'rpn_class': 0, 'rpn_box': 0, 'cls_class': 0, 'cls_box': 0}
        else:
            val_losses = {'rpn_loss': 0, 'class_loss': 0, 'mask_loss': 0}
        with torch.no_grad():
            for j, data in enumerate(tqdm(dataloader_val, disable=quiet, desc='Running Validation', file=sys.stdout)):               
                data = [x.to(device, non_blocking=True) for x in data]
                
                if is_fcn:
                    data.append(True)
                    val_loss_epoch, epoch_losses = model(*data)
                    for loss_type in val_losses.keys():
                        val_losses[loss_type] += epoch_losses[loss_type]
                else:
                    rpn_loss, class_loss, mask_loss = model(*data)
                    val_loss_epoch = rpn_loss + class_loss + mask_loss
                    val_losses['rpn_loss'] += rpn_loss.item()
                    val_losses['class_loss'] += class_loss.item()
                    val_losses['mask_loss'] += mask_loss.item()

                val_loss += val_loss_epoch.item()

                # compute validation mAP
                if is_fcn:
                    proposals, labels = model.evaluate(data[0], device=device)
                else:
                    proposals, labels, masks = model.evaluate(data[0], device=device)

                batch_truth_boxes += [entry.tolist() for entry in data[2]]
                batch_truth_labels += [entry.tolist() for entry in data[1]]
                batch_eval_boxes += [entry.tolist() for entry in proposals]
                batch_eval_labels += [entry.tolist() for entry in labels]
                
                if not is_fcn and j % mask_plot_freq == 0:
                    imgs, gt_labels, gt_bboxes, gt_masks = data
                    for k in range(imgs.shape[0]):
                        if len(proposals[k]) != 0:
                            MaskPlot.viz_mask(viz_dir, i, imgs[k], j*batch_size + k, masks[k], proposals[k], labels[k], gt_masks[k], gt_bboxes[k], gt_labels[k])
                
        val_mAP, ap_dict = EvalUtil.model_eval(id2str,
                                               batch_truth_boxes,
                                               batch_truth_labels,
                                               batch_eval_boxes,
                                               batch_eval_labels)
        val_loss = val_loss / len(dataloader_val)

        model.train()

        scheduler.step()

        # track loss
        loss /= len(dataloader)
        loss_tracker.append(loss)
        val_loss_tracker.append(val_loss)
        mAP_avg_tracker.append(val_mAP * 100)
        mAP_per_tracker.append(ap_dict)
        if verbose:
            print(ap_dict)
            print("  Training Loss: %.2f, Validation Loss %.2f, Validation mAP %.4f" % (loss, val_loss, val_mAP*100))

        # save the best model
        if (best_mAP is None) or (val_mAP > best_mAP):
            best_epoch = i
            best_mAP = val_mAP
            best_model = deepcopy(model.state_dict())

        # loss breakdown
        for loss_type in train_losses_tracker.keys():
            train_losses_tracker[loss_type].append(train_losses[loss_type] / len(dataloader))
            val_losses_tracker[loss_type].append(val_losses[loss_type] / len(dataloader_val))
        
        if save:
            model_filename = os.path.join(checkpoint_dir, f"model_{i}.pt")
            torch.save(model.state_dict(), model_filename)
            save_losses_curve(train_losses_tracker, val_losses_tracker, loss_dir, save_timestamp, i)
            
    if save:
        # save the model
        model_filename = os.path.join(base_dir, "model.pt")
        torch.save(model.state_dict(), model_filename)
        best_model_filename = os.path.join(base_dir, "model_epoch_{}.pt".format(best_epoch))
        torch.save(best_model, best_model_filename)

        # save the properties
        save_properties(model, optimizer, base_dir)

        # save the loss curve
        save_loss_curve(loss_tracker, val_loss_tracker, base_dir, save_timestamp)

        # save mAP information
        save_mAP(mAP_avg_tracker, mAP_per_tracker, base_dir, save_timestamp)

        # save loss breakdown
        save_losses_curve(train_losses_tracker, val_losses_tracker, loss_dir, save_timestamp, num_epochs)

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
    loss_curve_filename = os.path.join(base_dir, "loss_{}.png".format(save_timestamp))
    epochs = np.arange(1, len(loss_tracker) + 1, dtype=int).tolist()
    plt.cla()
    plt.plot(epochs, loss_tracker, label='Training')
    plt.plot(epochs, val_loss_tracker, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(loss_curve_filename)
    plt.cla()

    loss_filename = os.path.join(base_dir, "loss_{}_values.csv".format(save_timestamp))
    dataset = np.array([np.arange(1, len(loss_tracker) + 1).tolist(), loss_tracker, val_loss_tracker]).transpose().tolist()
    dataframe = pd.DataFrame(dataset, columns=['epoch', 'train_loss', 'val_loss'])
    dataframe.to_csv(loss_filename, index=False)


def save_mAP(mAP_avg_tracker, mAP_per_tracker, base_dir, save_timestamp):
    epochs = np.arange(1, len(mAP_avg_tracker) + 1, dtype=int).tolist()

    mAP_curve_filename = os.path.join(base_dir, "mAP_{}.png".format(save_timestamp))
    plt.cla()
    plt.plot(epochs, mAP_avg_tracker)
    plt.xlabel("Epoch")
    plt.ylabel("Validation mAP")
    plt.grid(True)
    plt.savefig(mAP_curve_filename)
    plt.cla()

    mAP_filename = os.path.join(base_dir, "mAP_{}_values.csv".format(save_timestamp))
    mAP_list = [epochs, mAP_avg_tracker]
    mAP_cols = ['epoch', 'avg']
    for key in mAP_per_tracker[0].keys():
        mAP_list.append([mAP_per_tracker[i][key] for i in range(len(mAP_per_tracker))])
        mAP_cols.append(key)
    dataset = np.array(mAP_list).transpose().tolist()
    dataframe = pd.DataFrame(dataset, columns=mAP_cols)
    dataframe.to_csv(mAP_filename, index=False)


def save_losses_curve(train_losses_tracker, val_losses_tracker, base_dir, save_timestamp, num_epochs):
    loss_curve_filename = os.path.join(base_dir, f"losses_{num_epochs}.png")
    epochs = np.arange(1, num_epochs + 1, dtype=int).tolist()
    plt.cla()
    data, cols = [], ['epoch']
    for loss_type in train_losses_tracker.keys():
        plt.plot(epochs, train_losses_tracker[loss_type], label="Train {}".format(loss_type))
        data.append(train_losses_tracker[loss_type])
        cols.append("train_{}_loss".format(loss_type))
    for loss_type in val_losses_tracker.keys():
        plt.plot(epochs, val_losses_tracker[loss_type], label="Val {}".format(loss_type))
        data.append(val_losses_tracker[loss_type])
        cols.append("val_{}_loss".format(loss_type))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(loss_curve_filename)
    plt.cla()

    loss_filename = os.path.join(base_dir, "losses_values.csv")
    dataset = np.array([np.arange(1, num_epochs + 1).tolist()] + data).transpose().tolist()
    dataframe = pd.DataFrame(dataset, columns=cols)
    dataframe.to_csv(loss_filename, index=False)
