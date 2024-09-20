import math
import random

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from torchvision import transforms
from tqdm import tqdm

    
def img_to_patch_ovelapped(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                        as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape

    # Calculate number of patches along width and height using ceil to cover every pixel
    num_patches_w = math.ceil(W / patch_size)
    num_patches_h = math.ceil(H / patch_size)

    # Calculate overlap for height and width
    # Overlap is calculated such that the entire image is covered with minimal overlap
    total_overlap_h = (num_patches_h * patch_size) - H
    total_overlap_w = (num_patches_w * patch_size) - W
    overlap_h = total_overlap_h // (num_patches_h - 1) if num_patches_h > 1 else 0
    overlap_w = total_overlap_w // (num_patches_w - 1) if num_patches_w > 1 else 0

    # Initialize an empty list to store patches
    patches = []

    # Extract patches
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            start_i = max(i * (patch_size - overlap_h), 0)
            start_j = max(j * (patch_size - overlap_w), 0)
            end_i = start_i + patch_size
            end_j = start_j + patch_size

            # Adjust for boundaries
            start_i -= max(end_i - H, 0)
            start_j -= max(end_j - W, 0)
            end_i = min(end_i, H)
            end_j = min(end_j, W)

            patch = x[:, :, start_i:end_i, start_j:end_j]

            patches.append(patch)

    # Stack patches into a tensor
    patches = torch.stack(patches, dim=1)  # [B, num_patches, C, patch_size, patch_size]
    
    if flatten_channels:
        patches = patches.view(B, -1, C * patch_size * patch_size)  # [B, num_patches, C*patch_size*patch_size]

    return patches


def get_top_k_patch_indices(patches, top_k_patches=3, patch_selection_criterion="contrast", patch_strategy="max"):
    if patch_strategy == "all":
        return list(range(len(patches)))
    elif patch_strategy == "random":
        return random.sample(range(len(patches)), top_k_patches)
    
    scores = [get_patch_complexity(p, patch_selection_criterion) for p in patches]
    scores = np.array(scores, dtype=np.int16)

    # Getting indices
    idx = np.argsort(scores)
    if patch_strategy == "max":
        idx = idx[::-1]
    
    idx = idx[:top_k_patches]

    return idx


def get_patch_complexity(patch, patch_selection_criterion="contrast"):
    # Calculate GLCM properties
    # print("patch", patch.shape)
    glcm = graycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    dissimilarity = graycoprops(glcm, patch_selection_criterion)[0, 0]

    return dissimilarity


def compute_patch_indices(filenames, patch_size, top_k_patches, patch_selection_criterion, patch_strategy="max"):
    transform_img = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Grayscale(),
    ])

    top_k_indices = {}
    for filename in tqdm(filenames):
        patches = img_to_patch_ovelapped(transform_img(Image.open(filename)).unsqueeze(0), patch_size, flatten_channels=False).squeeze()
        top_k_indices[filename] = get_top_k_patch_indices(
            patches,
            top_k_patches=top_k_patches,
            patch_selection_criterion=patch_selection_criterion,
            patch_strategy=patch_strategy,
        ).copy()

    return top_k_indices


def calculate_top_k_indices(filenames, patch_size, top_k_patches, patch_selection_criterion):
            
        transform_patch = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Grayscale(),
        ]
        )
        top_k_indices = {}
        for filename in tqdm(filenames):
            i1 = transform_patch(Image.open(filename).convert("RGB"))
            ps = extract_patches(i1, (patch_size, patch_size))
            ps = [p.squeeze() for p in ps]

            top_k_indices[filename] = get_top_k_patch_indices(ps, top_k_patches=top_k_patches, patch_selection_criterion=patch_selection_criterion).copy()
    
        return top_k_indices


def extract_patches(image_tensor, patch_size=(224, 224)):
    _, H, W = image_tensor.shape
    patch_height, patch_width = patch_size

    # Compute the number of patches along height and width
    num_patches_height = H // patch_height + (1 if H % patch_height != 0 else 0)
    num_patches_width = W // patch_width + (1 if W % patch_width != 0 else 0)

    for i in range(num_patches_height):
    
        for j in range(num_patches_width):
            # Compute the starting height and width of the patch
            start_height = i * patch_height
            start_width = j * patch_width

            # Compute the ending height and width of the patch
            end_height = start_height + patch_height
            end_width = start_width + patch_width

            if end_height > H:
                height_diff = end_height - H
                start_height -= height_diff
                end_height -= height_diff
            
            if end_width > W:
                width_diff = end_width - W
                start_width -= width_diff
                end_width -= width_diff

            # Extract the patch from the image tensor
            patch = image_tensor[:, start_height:end_height, start_width:end_width]

            # If this is the first patch, initialize the patches tensor
            if i == 0 and j == 0:
                patches = [patch]
            else:
                patches.append(patch)

    # Convert the list of patches to a single tensor
    patches_tensor = torch.stack(patches)
    
    return patches_tensor
