import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from sklearn.model_selection import train_test_split
import cv2
from typing import List, Tuple, Optional
import glob

class ProstateMRIDataset(Dataset):
    """Dataset for loading NIfTI files from organized folder structure"""
    
    def __init__(self, data_dir: str, data_type: str = 'train', 
                 transform=None, target_size: Tuple[int, int] = (128, 128),
                 use_segmentation: bool = False):
        """
        Args:
            data_dir: Root directory containing HipMRI_Study_open
            data_type: 'train', 'validate', or 'test'
            transform: Optional transforms
            target_size: Target image size for resizing
            use_segmentation: Whether to use segmentation masks
        """
        self.data_dir = data_dir
        self.data_type = data_type
        self.transform = transform
        self.target_size = target_size
        self.use_segmentation = use_segmentation
        
        # Build the correct path based on folder structure
        if use_segmentation:
            self.data_path = os.path.join(data_dir, f'keras_slices_seg_{data_type}')
        else:
            self.data_path = os.path.join(data_dir, f'keras_slices_{data_type}')
        
        print(f"Looking for data in: {self.data_path}")
        
        # Precompute all slices across files
        self.samples = self._precompute_slices()
    
    def _precompute_slices(self):
        """Precompute all slice indices across NIfTI files"""
        samples = []
        
        # Find all NIfTI files in the specific folder
        nifti_files = glob.glob(os.path.join(self.data_path, "*.nii.gz"))
        nifti_files.extend(glob.glob(os.path.join(self.data_path, "*.nii")))
        
        if not nifti_files:
            raise ValueError(f"No NIfTI files found in {self.data_path}")
        
        print(f"Found {len(nifti_files)} NIfTI files in {self.data_path}")
        
        for file_idx, file_path in enumerate(nifti_files):
            try:
                import nibabel as nib
                img = nib.load(file_path)
                data = img.get_fdata()
                
                # Determine number of slices
                if len(data.shape) == 3:
                    num_slices = data.shape[2]
                elif len(data.shape) == 4:
                    num_slices = data.shape[2]  # Third dimension is slices
                else:
                    num_slices = 1
                
                print(f"File {os.path.basename(file_path)}: {data.shape} -> {num_slices} slices")
                
                for slice_idx in range(num_slices):
                    samples.append({
                        'file_path': file_path,
                        'file_idx': file_idx,
                        'slice_idx': slice_idx
                    })
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Total slices for {self.data_type}: {len(samples)}")
        return samples
    
    def _load_slice(self, file_path: str, slice_idx: int) -> np.ndarray:
        """Load and preprocess a single slice from NIfTI file"""
        try:
            # Load NIfTI file
            import nibabel as nib
            img = nib.load(file_path)
            data = img.get_fdata()
            
            # Extract slice (NIfTI files are usually 3D: x, y, slices)
            if len(data.shape) == 3:
                slice_data = data[:, :, slice_idx]
            elif len(data.shape) == 4:  # 4D: x, y, slices, time/channels
                slice_data = data[:, :, slice_idx, 0]  # Take first channel
            else:
                slice_data = data  # Already 2D
            
            # Normalize to [0, 1]
            slice_min = np.min(slice_data)
            slice_max = np.max(slice_data)
            if slice_max > slice_min:
                slice_data = (slice_data - slice_min) / (slice_max - slice_min)
            else:
                slice_data = np.zeros_like(slice_data)
            
            # Resize to target size
            if slice_data.shape != self.target_size:
                slice_data = cv2.resize(slice_data, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Add channel dimension
            slice_data = np.expand_dims(slice_data, axis=0)
            
            return slice_data.astype(np.float32)
            
        except Exception as e:
            print(f"Error loading slice {slice_idx} from {file_path}: {e}")
            # Return blank image as fallback
            return np.zeros((1, *self.target_size), dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        slice_data = self._load_slice(sample_info['file_path'], sample_info['slice_idx'])
        
        # Convert to tensor
        slice_tensor = torch.from_numpy(slice_data)
        
        # Apply transforms if any
        if self.transform:
            slice_tensor = self.transform(slice_tensor)
        
        # Normalize to [-1, 1] for tanh activation
        slice_tensor = slice_tensor * 2 - 1
        
        return slice_tensor, sample_info

def create_data_loaders_keras(data_dir: str, batch_size: int = 16, 
                            target_size: Tuple[int, int] = (128, 128),
                            use_multimodal: bool = False):
    """Create data loaders using the organized folder structure"""
    
    if use_multimodal:
        # For multimodal, we need to handle both image and segmentation
        from torch.utils.data import Dataset as TorchDataset
        class CombinedDataset(TorchDataset):
            def __init__(self, data_dir, data_type, target_size):
                self.image_dataset = ProstateMRIDataset(data_dir, data_type, target_size=target_size, use_segmentation=False)
                self.seg_dataset = ProstateMRIDataset(data_dir, data_type, target_size=target_size, use_segmentation=True)
                self.samples = self.image_dataset.samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                image, image_info = self.image_dataset[idx]
                try:
                    seg_mask, _ = self.seg_dataset[idx]
                except:
                    seg_mask = torch.zeros_like(image)
                combined = torch.cat([image, seg_mask], dim=0)
                return combined, image_info
        
        DatasetClass = CombinedDataset
        in_channels = 2  # Image + segmentation
    else:
        DatasetClass = ProstateMRIDataset
        in_channels = 1
    
    # Create datasets using the organized folders
    train_dataset = DatasetClass(data_dir, 'train', target_size=target_size)
    val_dataset = DatasetClass(data_dir, 'validate', target_size=target_size)
    test_dataset = DatasetClass(data_dir, 'test', target_size=target_size)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, in_channels

def explore_h5_structure(data_dir: str):
    """Explore the folder structure and find actual files"""
    print(f"Exploring data directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory does not exist: {data_dir}")
        return
    
    # Check the main structure
    items = os.listdir(data_dir)
    print(f"Contents of {data_dir}:")
    for item in items:
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # Count ALL files in this directory
            all_files = os.listdir(item_path)
            nifti_files = [f for f in all_files if f.endswith('.nii.gz') or f.endswith('.nii')]
            other_files = [f for f in all_files if not f.endswith(('.nii.gz', '.nii'))]
            print(f"  📁 {item}/ - {len(nifti_files)} NIfTI files, {len(other_files)} other files")
            
            # Show first few NIfTI files
            for nifti_file in nifti_files[:3]:
                print(f"      📄 {nifti_file}")
            if len(nifti_files) > 3:
                print(f"      ... and {len(nifti_files) - 3} more")
        else:
            print(f"  📄 {item}")
    
    # Specifically check the train/val/test folders
    print(f"\n🔍 Checking specific data folders:")
    data_folders = [
        'keras_slices_train',
        'keras_slices_validate', 
        'keras_slices_test'
    ]
    
    for folder in data_folders:
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            nifti_files = glob.glob(os.path.join(folder_path, "*.nii.gz"))
            nifti_files.extend(glob.glob(os.path.join(folder_path, "*.nii")))
            print(f"  ✅ {folder}: {len(nifti_files)} NIfTI files")
            
            # Show what files are actually there
            if nifti_files:
                for f in nifti_files[:2]:
                    print(f"      {os.path.basename(f)}")
                if len(nifti_files) > 2:
                    print(f"      ... and {len(nifti_files) - 2} more")
            else:
                # Show what IS in the folder
                all_files = os.listdir(folder_path)
                print(f"      Folder contains: {all_files[:5]}...")
        else:
            print(f"  ❌ {folder}: NOT FOUND")

# For testing
if __name__ == "__main__":
    data_dir = "HipMRI_Study_open"
    explore_h5_structure(data_dir)