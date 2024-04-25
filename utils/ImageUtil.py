import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
from torchvision import ops


def build_image(image, bboxes, labels, box_color='y', show=False, filename=None):
    fig, ax = plt.subplots(figsize=(16, 8))

    # permute for matplotlib and add to ax
    image_permute = image.permute(1, 2, 0).cpu().numpy()
    ax.imshow(image_permute)

    if bboxes.nelement() != 0:

        # convert the bounding boxes back to xywh
        bboxes = ops.box_convert(bboxes, in_fmt='xyxy', out_fmt='xywh')

        # add the bounding boxes and the labels to ax
        for bbox, label in zip(bboxes, labels):
            # only display real labels
            if label == "pad":
                continue

            # display bounding box
            x, y, w, h = bbox.detach().cpu().numpy()
            rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)
            # display label
            ax.text(x + 5, y + 14, label, bbox=dict(facecolor='white', alpha=0.5))

    if show:
        fig.show()

    if filename is not None:
        fig.savefig(filename)


def build_grid_images(images, bboxes, labels, box_color='y', show=False, filename=None):
    if (len(images) % 3) != 0:
        raise ValueError("build_grid_images requires that len(images) be a multiple of 3, got {}".format(len(images)))
    num_rows = int(len(images) / 3)
    fig, axes = plt.subplots(num_rows, 3, figsize=(8, 9), layout="constrained")

    for i, (image, bboxes_i, labels_i) in enumerate(zip(images, bboxes, labels)):
        axes_idx = np.unravel_index(i, (num_rows, 3))

        # permute for matplotlib and add to ax
        image_permute = image.permute(1, 2, 0).cpu().numpy()
        axes[axes_idx].imshow(image_permute)
        axes[axes_idx].set_xticks([])
        axes[axes_idx].set_yticks([])
        axes[axes_idx].set_xticklabels([])
        axes[axes_idx].set_yticklabels([])

        if bboxes_i.nelement() == 0:
            continue

        # convert the bounding boxes back to xywh
        bboxes_i = ops.box_convert(bboxes_i, in_fmt='xyxy', out_fmt='xywh')

        # add the bounding boxes and the labels to ax
        for bbox, label in zip(bboxes_i, labels_i):
            # only display real labels
            if label == "pad":
                continue

            # display bounding box
            x, y, w, h = bbox.detach().cpu().numpy()
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=box_color, facecolor='none')
            axes[axes_idx].add_patch(rect)
            # display label
            axes[axes_idx].text(x + 10, y + 35, label, bbox=dict(facecolor='white', alpha=0.5))

    if show:
        fig.show()

    if filename is not None:
        fig.savefig(filename)
