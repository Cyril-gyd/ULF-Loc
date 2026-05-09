# from segment_anything import sam_model_registry, SamPredictor

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
from torchvision import transforms

import argparse
import os


def batched_nms(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2): 
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores

def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors

class SuperPoint(nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = 256

        self.nms_radius = 4
        self.max_num_keypoints = None
        self.detection_threshold = 0.00
        self.remove_borders = 4

        self.transform = transforms.Grayscale(num_output_channels=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, out_channels,
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)), strict=False)

        print('Loaded SuperPoint model')

    
    @torch.inference_mode()
    def detectAndCompute(self, x, top_k = None, detection_threshold = None):
        """ Compute sparse keypoints & descriptors. Supports batched mode."""
        self.max_num_keypoints = top_k
        
        device = next(self.conv1a.parameters()).device
        x = x.to(device)
        x = self.transform(x)
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        
        cDa = self.relu(self.convDa(x))
        descriptors_dense = self.convDb(cDa)
        descriptors_dense = torch.nn.functional.normalize(descriptors_dense, p=2, dim=1)
        
        scores = batched_nms(scores, self.nms_radius)

        # Discard keypoints near the image borders
        if self.remove_borders:
            pad = self.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        if b > 1:
            idxs = torch.where(scores > self.detection_threshold)
            mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
        else:  # Faster shortcut
            scores = scores.squeeze(0)
            idxs = torch.where(scores > self.detection_threshold)

        # Convert (i, j) to (x, y)
        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = scores[idxs]

        keypoints = []
        scores = []
        descriptors = []
        for i in range(b):
            if b > 1:
                k = keypoints_all[mask[i]]
                s = scores_all[mask[i]]
            else:
                k = keypoints_all
                s = scores_all
            if self.max_num_keypoints is not None:
                k, s = select_top_k_keypoints(k, s, self.max_num_keypoints)
            d = sample_descriptors(k[None], descriptors_dense[i, None], 8)
            keypoints.append(k)
            scores.append(s)
            descriptors.append(d.squeeze(0).transpose(0, 1))

        return [ {
            "keypoints": keypoints[i],
            "keypoint_scores": scores[i],
            "descriptors": descriptors[i], } for i in range(b)]
        
        
    @torch.inference_mode()
    def detectAndComputeDense(self, x):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        # print(x.shape)
        device = next(self.conv1a.parameters()).device
        x = x.to(device)
        x = self.transform(x)
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)

        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        return descriptors, scores.unsqueeze(1)
    
    def forward(self, x):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        # print(x.shape)
        x = self.transform(x)
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)

        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        return descriptors, scores


parser = argparse.ArgumentParser(
    description=(
        "Get image embeddings of an input image or directory of images."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where embeddings will be saved. Output will be either a folder "
        "of .pt per image or a single .pt representing image embeddings."
    ),
)


parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")


def main(args: argparse.Namespace) -> None:
    print("Loading model...")

    model = SuperPoint().cuda()

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        print(os.listdir(args.input))
        seqs = [f for f in os.listdir(args.input) if "seq" in f and "zip" not in f]
    
    print(seqs)
    os.makedirs(args.output, exist_ok=True)


    for seq in seqs:
        targets = [
            f"{seq}/{f}" for f in os.listdir(os.path.join(args.input, seq)) if "color" in f
        ]
        targets = [os.path.join(args.input, f) for f in targets]

        output_dir = os.path.join(args.output, seq)
        os.makedirs(output_dir, exist_ok=True)

        for t in targets:
            print(f"Processing '{t}'...")
            img_name = t.split(os.sep)[-1]
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            
            tensor_image = torch.from_numpy(np.array(image))
            input_image = tensor_image.to(
                device="cuda", dtype=torch.float32, non_blocking=True
            )
            input_image = input_image / 255.0
            img_features, scores = model(input_image.permute(2, 0, 1)[None])
            # print(scores.shape)

            img_features = img_features.squeeze(0) # (256, 60, 80)
            img_scores = scores[0] # (60, 80)

            torch.save(img_features, os.path.join(output_dir, f"{img_name}_fmap_CxHxW.pt"))
            torch.save(img_scores, os.path.join(output_dir, f"{img_name}_smap_CxHxW.pt"))
        

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

