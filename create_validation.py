#select random 80 images from train_images dir and move into validation_images dir
import os
import random
import shutil
import glob
import argparse
from tqdm import tqdm
import logging  

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_validation_set(train_dir, val_dir, num_images):
    """
    Create a validation set by moving a specified number of images from the training directory to the validation directory.

    Args:
        train_dir (str): Path to the training images directory.
        val_dir (str): Path to the validation images directory.
        num_images (int): Number of images to move from training to validation.
    """
    # Get all image files in the training directory
    image_files = glob.glob(os.path.join(train_dir, '*'))

    # Check if there are enough images in the training directory
    if len(image_files) < num_images:
        logger.error(f"Not enough images in {train_dir}. Found {len(image_files)}, but need {num_images}.")
        return

    # Select random images
    selected_images = random.sample(image_files, num_images)

    # Move selected images to the validation directory
    for image in tqdm(selected_images, desc="Moving images", unit="image"):
        shutil.move(image, val_dir)
    
    logger.info(f"Moved {num_images} images from {train_dir} to {val_dir}.")
def main():
    parser = argparse.ArgumentParser(description="Create a validation set from training images.")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to the training images directory.")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to the validation images directory.")
    parser.add_argument('--num_images', type=int, required=True, help="Number of images to move from training to validation.")

    args = parser.parse_args()

    # Create validation set
    create_validation_set(args.train_dir, args.val_dir, args.num_images)
if __name__ == "__main__":
    main()
# Example usage
# python3 create_validation.py --train_dir /path/to/train_images --val_dir /path/to/validation_images --num_images 80