import Augmentor
import numpy as np
from PIL import Image
import glob
from natsort import natsorted
import random
import matplotlib.pyplot as plt
import imageio
import os
import re

# Reading and sorting the image paths from the directories
# Replace paths with your own paths!!
ground_truth_images = natsorted(glob.glob("/Users/sroche/Desktop/augmented_output/*.png"))
mask_outline = natsorted(glob.glob("/Users/sroche/Desktop/augmented_output_outline/*.png"))
mask_images = natsorted(glob.glob("/Users/sroche/Desktop/augmented_output_mask/*.png"))

print(ground_truth_images)
print(mask_outline)

for i in range(0, len(ground_truth_images)):
    print("%s: Ground: %s | Mask 1: %s | Mask 2: %s" %
          (i+1, os.path.basename(ground_truth_images[i]),
           os.path.basename(mask_outline[i]),
           os.path.basename(mask_images[i])))

collated_images_and_masks = list(zip(ground_truth_images,
                                     mask_outline,
                                     mask_images))

collated_images_and_masks_length = len(collated_images_and_masks)

# Generate the list of alternating zeros and ones
y = [i % 2 for i in range(collated_images_and_masks_length)]

print(collated_images_and_masks)

images = [[np.asarray(Image.open(y)) for y in x] for x in collated_images_and_masks]

p = Augmentor.DataPipeline(images, y)

p.rotate(1, max_left_rotation=25, max_right_rotation=25)
p.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=10)
p.flip_top_bottom(0.5)
p.flip_left_right(0.5)
p.zoom_random(0.75, percentage_area=0.99)

# This line creates 1000 augmented images
augmented_images, labels = p.sample(1000)

r_index = random.randint(0, len(augmented_images)-1)
f, axarr = plt.subplots(1, 3, figsize=(20,15))
axarr[0].imshow(augmented_images[r_index][0])
axarr[1].imshow(augmented_images[r_index][1], cmap="gray")
axarr[2].imshow(augmented_images[r_index][2], cmap="gray")
plt.show()

# Output folders - replace with your own paths!
output_aug_folder = '/Users/sroche/Desktop/augmented_output/'
output_outline_folder = '/Users/sroche/Desktop/augmented_output_outline/'
output_mask_folder = '/Users/sroche/Desktop/augmented_output_mask/'

# Iterate over each image and save it
# Note, 210 was used due to adding the above outputs to folders that already contained data.
# You may want to simply delete the 210 if you are testing this code.
for i, image in enumerate(augmented_images):
    # Define the file path
    file_name1 = f"aug_img_{i+210}.png"  # You can change the file naming scheme as needed
    file_name2 = f"aug_outline_{i+210}.png"
    file_name3 = f"aug_mask_{i+210}.png"
    file_path1 = os.path.join(output_aug_folder, file_name1)
    file_path2 = os.path.join(output_outline_folder, file_name2)
    file_path3 = os.path.join(output_mask_folder, file_name3)

    # Save the image
    imageio.imwrite(file_path1, augmented_images[i][0])
    imageio.imwrite(file_path2, augmented_images[i][1])
    imageio.imwrite(file_path3, augmented_images[i][2])

# This following code is optional!!
# It was used to rename some of the mask files that were named improperly


def rename_masks():
    # Directory containing the files
    directory = '/Users/sroche/Desktop/things_to_move'

    # Regular expression pattern to match file names
    pattern = r'aug_img_mask_(\d+)\.png'

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        # Check if the file matches the pattern
        match = re.match(pattern, filename)
        if match:
            # Extract the integer part from the file name
            number = match.group(1)
            # Construct the new file name
            new_filename = f"aug_img_{number}_mask.png"
            # Rename the file
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
