# ======================================================================================
# AI Declaration
#
# This script was developed with the assistance of an AI language model.
#
# Tasks performed by the AI:
# - Generated the boilerplate structure for the PyTorch `Dataset` and `DataLoader`.
# - Implemented the logic for loading NIfTI files using the `nibabel` library.
# Prompt:
# "Write a PyTorch Dataset class for NIfTI files (.nii.gz). It should load an image
# using nibabel, convert it to a tensor, and apply a transformation pipeline to resize
# it to 128x128 and normalize it to the range [-1, 1]. The __getitem__ method should
# return the image tensor, its full filename, and a 'subject label' extracted from the
# first two parts of the filename."
#
# LLM Used: Gemini 2.5 Pro
# ======================================================================================

"""
Data loading and preprocessing for the HipMRI 2D slices dataset.
- Loads Nifti files.
- Resizes images to a consistent size specified in the config.
- Normalizes images to the range [-1, 1] for VQ-VAE training.
"""
import os
import glob
import torch
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from config import cfg
import matplotlib.pyplot as plt
class HipMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        """Initializes the dataset object.

        Args:
            root_dir (str): Path to the root directory containing the dataset splits
                            (e.g., 'keras_slices_data/').
            transform (callable, optional): A function/transform to apply to each image tensor.
                                            Defaults to None.
            split (str, optional): The dataset split to load. Must be one of 'train',
                                   'validate', or 'test'. Defaults to 'train'.
        """
        self.image_dir = os.path.join(root_dir, f'keras_slices_{split}')
        self.transform = transform
        
        if not os.path.isdir(self.image_dir):
            raise ValueError(f"Data directory not found at: {self.image_dir}")
            
        self.image_files = [Path(p) for p in glob.glob(os.path.join(self.image_dir, '*.nii.gz'))]
        
        if not self.image_files:
            raise ValueError(f"No '.nii.gz' files found in {self.image_dir}")

        print(f"Found {len(self.image_files)} images in '{split}' set.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Fetches the sample at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The preprocessed image tensor.
                - str: The filename of the image.
                - str: The extracted subject ID label.
        """
        img_path = self.image_files[idx]
        
        try:
            nifti_img = nib.load(img_path)
            image = nifti_img.get_fdata().astype('float32')
        except Exception as e:
            # Provide dummy values for all three return items on error
            print(f"Error loading file: {img_path}. Skipping. Error: {e}")
            return torch.zeros((cfg.IN_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)), "error.nii.gz", "error"

        image_tensor = torch.from_numpy(image).unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        # --- NEW: Extract the subject ID as the "class label" ---
        # Assumes filename format like 'OAS1_0001_MR1_55.nii.gz'
        try:
            subject_label = "_".join(img_path.name.split('_')[:2])
        except IndexError:
            subject_label = "unknown"

        # --- KEY CHANGE: Return image, full filename, AND the subject label ---
        return image_tensor, img_path.name, subject_label

def get_dataloaders():
    """Creates and returns the data loaders for all dataset splits.

    This function initializes the train, validation, and test datasets and wraps them
    in PyTorch DataLoader objects with configurations specified in `config.py`.

    Returns:
        tuple: A tuple containing:
            - DataLoader: The data loader for the training set.
            - DataLoader: The data loader for the validation set.
            - DataLoader: The data loader for the test set.
    """

    transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), antialias=True),
        transforms.Lambda(lambda x: (x / x.max()) * 2.0 - 1.0),
    ])

    # --- CREATE ALL THREE DATASETS ---
    train_dataset = HipMRIDataset(root_dir=cfg.DATASET_ROOT, transform=transform, split='train')
    val_dataset = HipMRIDataset(root_dir=cfg.DATASET_ROOT, transform=transform, split='validate')
    test_dataset = HipMRIDataset(root_dir=cfg.DATASET_ROOT, transform=transform, split='test')
    
    # --- CREATE ALL THREE DATALOADERS ---
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, # Shuffle is False for testing
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    
    # Return all three loaders
    return train_loader, val_loader, test_loader


# Sanity Check 
# can run this file directly on Rangpur to test the data loading
if __name__ == '__main__':
    print("Running dataset sanity check...")
    try:
        train_loader, val_loader = get_dataloaders()
        print("Successfully created DataLoaders.")

        # Get a single batch from the training loader
        sample_batch = next(iter(train_loader))
        
        print("\n--- Training Batch ---")
        print(f"Batch shape: {sample_batch.shape}")
        print(f"Data type: {sample_batch.dtype}")
        print(f"Min value in batch: {sample_batch.min():.4f}")
        print(f"Max value in batch: {sample_batch.max():.4f}")
        
        # Save the first image of the batch to a file for visual inspection
        first_image = sample_batch[0].squeeze().cpu().numpy()
        plt.imshow(first_image, cmap='gray')
        plt.title("Sample Preprocessed Image")
        plt.colorbar()
        plt.savefig("sample_preprocessed_image.png")
        print("\nSaved a sample preprocessed image to 'sample_preprocessed_image.png'")
        
    except (ValueError, FileNotFoundError) as e:
        print(f"\nERROR: Could not run sanity check. {e}")
        print("Please ensure you are running this on a machine with access to the dataset path in config.py.")