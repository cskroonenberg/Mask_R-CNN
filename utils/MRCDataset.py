import albumentations
import numpy as np
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
import cv2

"""
corresponding bounding box scaling is taken from: https://sheldonsebastian94.medium.com/resizing-image-and-bounding-boxes-for-object-detection-7b9d9463125a
"""


class MRCDataset(Dataset):
    def __init__(self, dataset, img_size, str2id, dataset_type):
        """
        :type dataset: fiftyone.core.dataset.Dataset
        """

        self.dataset_type = dataset_type
        self.n_samples = None
        self.labels, self.bboxes, self.paths = [], [], []
        self.img_size = img_size
        self.max_mask_dim = [0,0,0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parse_dataset(dataset, img_size, str2id)
        self.dataset = dataset
        

    def paint_mask(self, image, bbox, mask):
        mask_color = np.array([np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255)], dtype='uint8')

        x_min, y_min, x_max, y_max = bbox
        mask_image = image.copy()

        mask_image[y_min:y_min+mask.shape[0], x_min:x_min+mask.shape[1]][mask] = mask_color

        return cv2.addWeighted(mask_image, 0.4, image, 0.6, 0.0)

    def parse_dataset(self, dataset, img_size, str2id):
        # create resize transform pipeline
        h_out, w_out = img_size
        transform = albumentations.Compose([albumentations.Resize(height=h_out, width=w_out, always_apply=True)],
                                           bbox_params=albumentations.BboxParams(format='pascal_voc'))

        for image_id, file_path in enumerate(tqdm(dataset.values("filepath"), desc="Pre-processing [{}] Dataset".format(self.dataset_type))):
            # load the image data and convert it to a tensor
            try:
                pil_img = Image.open(file_path).convert("RGB")
            except OSError:
                print(f"OSError loading {file_path}. Skipping")
                continue
            img_data = pil_to_tensor(pil_img)
            _, h_img, w_img = img_data.shape

            # load the sample
            sample = dataset[file_path]

            # parse detections
            labels_i, bboxes_i = [], []
            if sample["ground_truth"] is None:
                continue
            for i, detection in enumerate(sample["ground_truth"].detections):
                x_min, y_min, w, h = detection.bounding_box
                x_min, w = x_min * w_img, w * w_img
                y_min, h = y_min * h_img, h * h_img
                x_max, y_max = x_min + w, y_min + h
                label = str2id[detection.label]
                labels_i.append(label)
                bboxes_i.append([x_min, y_min, x_max, y_max, label])

                # masks_i.append(torch.tensor(detection.mask).type('torch.CharTensor'))
                
                self.max_mask_dim[1] = max(detection.mask.shape[0], self.max_mask_dim[1]) 
                self.max_mask_dim[2] = max(detection.mask.shape[1], self.max_mask_dim[2]) 
            
            self.max_mask_dim[0] = max(len(labels_i), self.max_mask_dim[0] ) 
                
            # perform transformations for re-sizing
            try:
                transformed = transform(image=img_data[0, :].cpu().numpy(), bboxes=np.array(bboxes_i))
            except ValueError as e:
                print(e)
                print("Skipping...")
                continue

            self.labels.append(torch.tensor(labels_i))
                                
            transformed = transform(image=img_data[0, :].cpu().numpy(), bboxes=np.array(bboxes_i))
            tboxes = torch.tensor(transformed['bboxes'])[:, :-1]
            self.bboxes.append(tboxes)
            self.paths.append(file_path)

        # store the dataset information
        self.n_samples = len(self.labels)
        self.labels = pad_sequence(self.labels, batch_first=True, padding_value=-1)
        self.bboxes = pad_sequence(self.bboxes, batch_first=True, padding_value=-1)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        sample = self.dataset[self.paths[index]]

        masks = []
        for i, detection in enumerate(sample["ground_truth"].detections):
            masks.append(torch.tensor(detection.mask).type('torch.CharTensor'))
    
        pil_img = Image.open(self.paths[index]).convert("RGB")
        img = pil_to_tensor(pil_img.resize(self.img_size[::-1])).type(torch.float32)
        return img, self.labels[index], self.bboxes[index], self.pad_masks(masks).type(torch.float32)
    
    def pad_masks(self, masks):
        padded_masks = [
            F.pad(mask, (0, self.max_mask_dim[2] - mask.shape[1], 0, self.max_mask_dim[1] - mask.shape[0]), mode='constant', value=-1)
            for mask in masks 
        ]
        for i in range(self.max_mask_dim[0] - len(masks)):
            padded_masks.append(torch.full((self.max_mask_dim[1], self.max_mask_dim[2]), -1))
            
        return torch.stack(padded_masks) #.to(torch.float32)