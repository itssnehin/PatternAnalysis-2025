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
import matplotlib.pyplot as plt

# Import the configuration object
from config import cfg

class HipMRIDataset(Dataset):
    """
    Custom PyTorch Dataset for loading HipMRI 2D slices.
    """
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (str): Path to the 'keras_slices_data' directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (str): One of 'train', 'validate', or 'test'.
        """
        # --- MODIFIED LOGIC TO HANDLE ALL SPLITS ---
        if split == 'train':
            self.image_dir = os.path.join(root_dir, 'keras_slices_train')
        elif split == 'validate':
            self.image_dir = os.path.join(root_dir, 'keras_slices_validate')
        elif split == 'test':
            self.image_dir = os.path.join(root_dir, 'keras_slices_test')
        else:
            raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'validate', 'test'.")
            
        self.transform = transform
        
        if not os.path.isdir(self.image_dir):
            raise ValueError(f"Data directory not found at: {self.image_dir}")
            
        self.image_files = glob.glob(os.path.join(self.image_dir, '*.nii.gz'))
        
        if not self.image_files:
            raise ValueError(f"No '.nii.gz' files found in {self.image_dir}")

        print(f"Found {len(self.image_files)} images in '{split}' set.")


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        try:
            # Load the Nifti image and get its data array
            nifti_img = nib.load(img_path)
            image = nifti_img.get_fdata().astype('float32')
        except Exception as e:
            print(f"Error loading file: {img_path}. Skipping. Error: {e}")
            # Return a dummy tensor if loading fails
            return torch.zeros((cfg.IN_CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

        # Convert numpy array to PyTorch tensor and add a channel dimension
        # Shape becomes [1, H, W]
        image_tensor = torch.from_numpy(image).unsqueeze(0)

        # Apply transformations if they exist
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor

def get_dataloaders():
    """
    Creates and returns the training, validation, AND testing DataLoaders.
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


# --- Sanity Check ---
# You can run this file directly on Rangpur to test the data loading
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