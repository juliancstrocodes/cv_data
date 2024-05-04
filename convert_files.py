import numpy as np
import imageio
from scipy.ndimage import label, binary_opening, generate_binary_structure
import matplotlib.pyplot as plt
import re
import os
from cellpose import io, dynamics
from PIL import Image

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


def process_images_and_masks(folder):
    # Process each file in the folder
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        with Image.open(file_path) as img:
            if 'mask' in filename:
                # Assuming that filenames containing 'mask' are mask files
                if img.mode == 'I':  # Check if mask is in 'I' 32-bit integer mode
                    # Convert 'I' mode to 'L' mode, scaling down values if necessary
                    img_array = np.array(img, dtype=np.uint32)
                    max_val = np.max(img_array)
                    if max_val > 0:  # Avoid division by zero
                        img_array = (img_array / max_val) * 255
                    img_8bit = Image.fromarray(img_array.astype(np.uint8), 'L')
                    img_8bit.save(file_path)  # Overwrite the original file
                    print(f"Converted and overwritten mask: {file_path}")
            else:
                # Convert RGBA images to grayscale
                if img.mode == 'RGBA':
                    gray_img = img.convert('L')  # Convert to 8-bit grayscale
                    gray_img.save(file_path)  # Overwrite the original file
                    print(f"Converted and overwritten image: {file_path}")


def convert_masks_to_tiff(input_folder, mask_identifier='mask'):
    # Get all PNG files in the input directory
    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    for file in files:
        if mask_identifier in file:
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(input_folder, file.replace('.png', '.tif'))

            # Read the PNG mask
            mask = imageio.imread(input_path)

            # Convert to TIFF and save
            imageio.imwrite(output_path, mask, format='TIFF')
            
            # Remove the original PNG file
            os.remove(input_path)
            print(f"Converted and replaced {input_path} with {output_path}")

# does not create a cellpose compatible npy file
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

def check_bit_depth(input_folder):
    # Get all image files in the input directory
    files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.tif')]
    
    for file in files:
        path = os.path.join(input_folder, file)
        with Image.open(path) as img:
            print(f"{file}: {img.mode}, {img.getbands()}")



dir_path = './cv_data/cleaned_augmented_data'
mask_filter = '_mask'

# Collect all images and masks
images = [f for f in os.listdir(dir_path) if not f.endswith(mask_filter + '.png')]
masks = [f for f in os.listdir(dir_path) if f.endswith(mask_filter + '.png')]

# Check for images without a corresponding mask
for img in images:
    expected_mask = img.replace('.png', mask_filter + '.png')
    if expected_mask not in masks:
        print(f"Missing mask for image: {img}")

# Check for masks without a corresponding image
for mask in masks:
    expected_image = mask.replace(mask_filter + '.png', '.png')
    if expected_image not in images:
        print(f"Missing image for mask: {mask}")

# input_folder = './cv_data/cleaned_augmented_data'
# process_images_and_masks(input_folder)


# process_all_masks(input_folder, input_folder, min_size=100)

# convert_masks_to_tiff(input_folder)

# input_folder = 'C:\\Users\\Gamer\\Desktop\\local_augmented_data'
# process_all_masks(input_folder, input_folder, min_size=100)


# img_path = 'C:\\Users\\Gamer\\Desktop\\local_augmented_data\\aug_img_3.png'
# mask_path = 'C:\\Users\\Gamer\\Desktop\\local_augmented_data\\aug_img_3_mask.png'
# convert_to_npy(img_path, mask_path)