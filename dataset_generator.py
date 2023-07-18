import cv2
import os

def extract_frames(video_path, output_path, target_shape=(64, 64, 3)):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Read the video file
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        # Read the next frame
        ret, frame = video.read()

        if not ret:
            break

        # Resize the frame to the target shape
        resized_frame = cv2.resize(frame, target_shape[:2])

        # Save the frame as an image
        frame_filename = f"frame_{frame_count}8.jpg"
        frame_path = os.path.join(output_path, frame_filename)
        cv2.imwrite(frame_path, resized_frame)

        frame_count += 1

    # Release the video capture object
    video.release()

    print(f"Frames extracted: {frame_count}")
    print(f"Output directory: {output_path}")


# Example usage
video_file = "D:/Python Projects/Leather defect detection/Leather Detection/Input_Videos/videos (8).mp4"
output_directory = "D:/Python Projects/Leather defect detection/Leather Detection/Output_dataset"
target_shape = (64, 64, 3)
extract_frames(video_file, output_directory, target_shape)

