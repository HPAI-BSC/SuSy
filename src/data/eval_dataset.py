import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

from utils.image_utils import calculate_top_k_indices, extract_patches

class EvalDataset(Dataset):
    def __init__(self, patch_size, top_k_patches, patch_selection_criterion, folder_path, transform_eval, use_center_crop=False):
        super().__init__()

        self.patch_size = patch_size
        self.top_k_patches = top_k_patches
        self.patch_selection_criterion = patch_selection_criterion

        self.filenames = []

        self.transform_eval = transform_eval
        self.patches = {}
        self.use_center_crop = use_center_crop

        folder_path = Path(folder_path)
        for frame in sorted(folder_path.iterdir()):
            if frame.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                im = Image.open(frame)
                if im.size[0] < 224 or im.size[1] < 224: # Min 224 x 224
                    continue
                self.filenames.append(frame)

        if not self.use_center_crop:
            self.top_k_indices = self._compute_top_k_indices()
        else:
            self.top_k_indices = None

        print(f"Loaded {len(self.filenames)} images")

    def _compute_top_k_indices(self):

        top_k_indices = dict()
        if len(self.filenames) > 0:
            print(f"Computing top k indices for {len(self.filenames)} missing filenames")
            top_k_indices = calculate_top_k_indices(self.filenames, self.patch_size, self.top_k_patches, self.patch_selection_criterion)
  
        return top_k_indices

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.filenames[idx]).convert("RGB"), dtype=np.float32)
        image = image / 255.0

        image = self.transform_eval(image=image)["image"]

        # if 4 channels, remove the alpha channel
        if image.shape[0] == 4:
            image = image[:3, :, :]

        if self.use_center_crop:
            return image
        
        patches = extract_patches(image, patch_size=(self.patch_size, self.patch_size))

        if self.top_k_patches > 0:
            top_k = self.top_k_indices[self.filenames[idx]][:self.top_k_patches]
        elif self.top_k_patches < 0:
            top_k = self.top_k_indices[self.filenames[idx]][self.top_k_patches:]
        else:
            raise ValueError("Top k indices cannot be 0")
        
        image = image.unsqueeze(0)
        best_patches = patches[top_k, :]

        return best_patches
