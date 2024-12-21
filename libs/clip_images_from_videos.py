import os
import re

import cv2

from libs.split_images import split_images_into_parts
from libs.variables import *


def extract_frames_from_videos(input_folder, output_folder, frame_interval=30):
    """
    Extract frames from all videos in a folder and save them to a specified directory.

    Args:
        input_folder (str): Path to the folder containing the videos.
        output_folder (str): Path to the folder where frames will be saved.
        frame_interval (int): Save every `frame_interval`-th frame. Default is 30.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    saved_frame_count = 0

    # Loop through all files in the input folder
    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)

        # Check if the file is a video
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Skipping non-video file: {video_file}")
            continue

        video_output_dir = output_folder

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_file}")
            continue

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:  # End of the video
                break

            # Save the frame at the given interval
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(video_output_dir, f"frame_{saved_frame_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1

            frame_count += 1

        cap.release()
        print(f"Extracted {saved_frame_count} frames from {video_file} and saved to {video_output_dir}")

    print("All videos processed!")

if __name__ == '__main__':
    # Example usage
    train_data_path = os.path.join(os.getcwd(), 'data', 'original_train_images')

    frame_interval = 30  # Extract one frame every 30 frames (adjust as needed)

    # extract_frames_from_videos(videos_save_path, train_data_path, frame_interval)

    split_images_path = os.path.join(os.getcwd(), 'data', 'split_images')

    split_images_into_parts(train_data_path, split_images_path, clip_width, clip_height)

