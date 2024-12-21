import os
from PIL import Image

from libs.variables import *


def split_images_into_parts(source_folder, output_folder, w, h):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'JPG'))]

    for image_file in image_files:
        # Load the image
        image_path = os.path.join(source_folder, image_file)
        image = Image.open(image_path)
        width, height = image.size

        rows = height // h
        columns = width // w

        h = height // rows
        w = width // columns

        # Split the image into 16 parts
        for i in range(rows):  # Rows
            for j in range(columns):  # Columns
                # Calculate the coordinates of the part
                left = j * w
                upper = i * h
                right = (j + 1) * w
                lower = (i + 1) * h

                # Crop the part from the image
                part = image.crop((left, upper, right, lower))

                # Save the part to the output folder
                part_filename = f"{os.path.splitext(image_file)[0]}_part_{i}_{j}.png"
                part.save(os.path.join(output_folder, part_filename))

    print(f"Images split into 400 parts and saved to '{output_folder}'.")


if __name__ == '__main__':
    # Example Usage
    # source_folder = r"D:\files\phtographing\train_images"  # Replace with the path to your folder with images
    source_folder = kaggle_data_path  # Replace with the path to your folder with images
    # output_folder = os.path.join(os.path.dirname(source_folder), "split_images")  # Creates a sibling folder
    output_folder = os.path.join(os.getcwd(), 'models')
    split_images_into_parts(source_folder, output_folder, clip_width, clip_height)
