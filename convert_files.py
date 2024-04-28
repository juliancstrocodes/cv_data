import numpy as np
import imageio
from scipy.ndimage import label, binary_opening, generate_binary_structure
import matplotlib.pyplot as plt
import re
import os
from cellpose import io, dynamics

def process_single_mask(input_path, output_path, min_size=100):
    mask = imageio.imread(input_path)
    
    # Normalize mask
    mask = mask > 0

    # Clean mask using binary opening (to remove small objects/noise)
    structure = generate_binary_structure(2, 2)  # Define structure for morphological operation
    clean_mask = binary_opening(mask, structure=structure)
    
    # Label the objects
    labeled_mask, num_features = label(clean_mask)

    # Remove small objects
    component_sizes = np.bincount(labeled_mask.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[labeled_mask]
    labeled_mask[too_small_mask] = 0

    # Relabel after removing small objects
    labeled_mask, num_features = label(labeled_mask > 0)

    # Map the labeled mask to 0 - 255 range for visualization as PNG. But this is not needed since 
    # mask bit depth on train datasets have 16-bit depth
    #labeled_mask_png = (255 * labeled_mask / np.max(labeled_mask)).astype(np.uint8) if np.max(labeled_mask) > 0 else np.zeros_like(labeled_mask)

    plt.figure(figsize=(8, 8))
    plt.imshow(labeled_mask, cmap='jet', interpolation='none')
    plt.colorbar()
    plt.title(f'Number of features after cleaning: {num_features}')
    plt.show()

    imageio.imwrite(output_path, labeled_mask)
    print(f"Labeled mask saved to: {output_path}")

def process_all_masks(input_folder, output_folder, min_size=100, file_pattern=r'aug_img_\d+_mask\.png'):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Compile the regex pattern for names
    pattern = re.compile(file_pattern)

    mask_files = [f for f in os.listdir(input_folder) if pattern.match(f)]
    for mask_file in mask_files:
        input_path = os.path.join(input_folder, mask_file)
        output_path = os.path.join(output_folder, mask_file)

        mask = imageio.imread(input_path)

        # Normalize mask
        mask = mask > 0

        structure = generate_binary_structure(2, 2)
        clean_mask = binary_opening(mask, structure=structure)
        labeled_mask, _ = label(clean_mask)

        component_sizes = np.bincount(labeled_mask.ravel())
        too_small = component_sizes < min_size
        too_small_mask = too_small[labeled_mask]
        labeled_mask[too_small_mask] = 0
        labeled_mask, _ = label(labeled_mask > 0)
        
        imageio.imwrite(output_path, labeled_mask)
        print(f"Processed and saved: {output_path}")

def convert_single_mask_to_npy(png_path, output_folder):
    # Extract the base file name without extension
    base_file_name = re.sub(r'_mask\.png$', '', os.path.basename(png_path))
    
    # Create the .npy file name
    npy_file_name = f"{base_file_name}_seg.npy"
    npy_path = os.path.join(output_folder, npy_file_name)

    # Load the PNG image as a numpy array
    mask_png = imageio.imread(png_path)
    
    # Convert any non-zero values to the object ID (1, 2, 3, ...)
    # Assuming the labeled mask PNG is already in the correct labeled format
    mask_npy = mask_png.astype(np.uint16)  # Ensure dtype is int, with enough range

    # Save the numpy array as a .npy file
    np.save(npy_path, mask_npy)
    print(f"Converted and saved: {npy_path}")


# Unsure if correct. Need to load png masks to compatible format for cellpose seg.npy expectations
def load_masks(png_path):
    masks = imageio.imread(png_path)
    unique_values = np.unique(masks)
    if unique_values[0] != 0:
        print("Error: masks not zero based")
        return None
    
    return masks

# DOESN'T work. Can't figure out proper format for masks and flows input
def convert_to_npy(img_path, mask_path, file_pattern=r'aug_img_\d+_mask\.png'):
    image = imageio.imread(img_path)
    masks = load_masks(mask_path)
    print(masks)
    flows = dynamics.labels_to_flows(masks)
    io.masks_flows_to_seg(image, masks, flows, img_path)

def check_npy_file(npy_path):
    try:
        data = np.load(npy_path,allow_pickle=True)
        print(f"File {npy_path} info:")
        print(f" - shape: {data.shape}")
        print(f" - dtype: {data.dtype}")
        print(f" - unique values: {np.unique(data)}")
    except Exception as e:
        print(f" - ERROR loading {npy_path}: {e}")


dir_path = './test'
mask_filter = '_mask'
images = [f for f in os.listdir(dir_path) if not f.endswith(mask_filter + '.png')]
masks = [f for f in os.listdir(dir_path) if f.endswith(mask_filter + '.png')]

for img in images:
    expected_mask = img.replace('.png', mask_filter + '.png')
    if expected_mask not in masks:
        print(f"Missing mask for image: {img}")
# input_folder = 'C:\\Users\\Gamer\\Desktop\\local_augmented_data'
# process_all_masks(input_folder, input_folder, min_size=100)


# img_path = 'C:\\Users\\Gamer\\Desktop\\local_augmented_data\\aug_img_3.png'
# mask_path = 'C:\\Users\\Gamer\\Desktop\\local_augmented_data\\aug_img_3_mask.png'
# convert_to_npy(img_path, mask_path)