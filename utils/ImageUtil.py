from matplotlib import patches
import matplotlib.pyplot as plt
from torchvision import ops


def build_image(image, bboxes, labels, box_color='y', show=False, filename=None):
    fig, ax = plt.subplots(figsize=(16, 8))

    # permute for matplotlib and add to ax
    image_permute = image.permute(1, 2, 0).cpu().numpy()
    ax.imshow(image_permute)

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
