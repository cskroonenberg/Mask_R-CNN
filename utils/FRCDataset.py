import albumentations
import numpy as np
from PIL import Image
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

"""
corresponding bounding box scaling is taken from: https://sheldonsebastian94.medium.com/resizing-image-and-bounding-boxes-for-object-detection-7b9d9463125a
"""


class FRCDataset(Dataset):
    def __init__(self, dataset, img_size, str2id, dataset_type):
        """
        :type dataset: fiftyone.core.dataset.Dataset
        """

        self.dataset_type = dataset_type
        self.n_samples = None
        self.images = None
        self.labels = None
        self.bboxes = None
        self.parse_dataset(dataset, img_size, str2id)

    def parse_dataset(self, dataset, img_size, str2id):

        # create resize transform pipeline
        h_out, w_out = img_size
        transform = albumentations.Compose([albumentations.Resize(height=h_out, width=w_out, always_apply=True)],
                                           bbox_params=albumentations.BboxParams(format='pascal_voc'))

        images, labels, bboxes = [], [], []
        for image_id, file_path in enumerate(tqdm(dataset.values("filepath"), desc="Pre-processing [{}] Dataset".format(self.dataset_type))):
            # load the image data and convert it to a tensor
            pil_img = Image.open(file_path).convert("RGB")
            img_data = pil_to_tensor(pil_img).type(torch.float32)
            _, h_img, w_img = img_data.shape

            # load the sample
            sample = dataset[file_path]

            # parse out the detections
            labels_i, bboxes_i = [], []
            for detection in sample["ground_truth"].detections:
                x_min, y_min, w, h = detection.bounding_box
                x_min, w = x_min * w_img, w * w_img
                y_min, h = y_min * h_img, h * h_img
                x_max, y_max = x_min + w, y_min + h
                label = str2id[detection.label]
                labels_i.append(label)
                bboxes_i.append([x_min, y_min, x_max, y_max, label])

            # perform transformations for re-sizing
            transformed = transform(image=img_data[0, :].cpu().numpy(), bboxes=np.array(bboxes_i))
            img_data = pil_to_tensor(pil_img.resize(img_size[::-1])).type(torch.float32).cpu()

            images.append(img_data)
            labels.append(torch.tensor(labels_i))
            bboxes.append(torch.tensor(transformed['bboxes'])[:, :-1])

        # store the dataset information
        self.n_samples = len(labels)
        self.images = torch.stack(images).cpu()
        self.labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        self.bboxes = pad_sequence(bboxes, batch_first=True, padding_value=-1)

    def batches(self, batch_size):
        # shuffle the data order
        idxs = list(range(self.n_samples))
        idxs = random.sample(idxs, len(idxs))

        start_idx, end_idx = 0, 0
        images_batches, labels_batches, bboxes_batches = [], [], []
        while (start_idx < self.n_samples) and (end_idx < self.n_samples):
            # determine the indices for this batch
            end_idx = start_idx + batch_size
            if (end_idx + batch_size) > self.n_samples:
                end_idx = self.n_samples
            batch_idxs = idxs[start_idx:end_idx]

            images_batches.append(self.images[batch_idxs].clone().cpu())
            labels_batches.append(self.labels[batch_idxs].clone())
            bboxes_batches.append(self.bboxes[batch_idxs].clone())
            start_idx += batch_size

        return images_batches, labels_batches, bboxes_batches
