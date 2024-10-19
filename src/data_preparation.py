import cv2
import os
import shutil
import random
from ultralytics import YOLO
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_interval=30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"Extracted {saved_count} frames from {video_path}")

def annotate_image(model, image_path, label_path):
    results = model(image_path)
    
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    with open(label_path, 'w') as f:
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 2:  # YOLO class 2 is typically 'car'
                    x, y, w, h = box.xywh[0]
                    # Convert to YOLO format (normalized coordinates)
                    x_center = x / width
                    y_center = y / height
                    w = w / width
                    h = h / height
                    f.write(f"0 {x_center} {y_center} {w} {h}\n")

def prepare_dataset(videos_dir, output_dir, train_ratio=0.8):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(os.path.join(images_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(images_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, "val"), exist_ok=True)

    all_frames = []

    for video_file in os.listdir(videos_dir):
        if video_file.endswith((".mp4", ".avi", ".mov")):
            video_name = os.path.splitext(video_file)[0]
            video_path = os.path.join(videos_dir, video_file)
            frames_dir = os.path.join(output_dir, f"temp_frames_{video_name}")
            
            extract_frames(video_path, frames_dir)

            for frame_file in os.listdir(frames_dir):
                all_frames.append((os.path.join(frames_dir, frame_file), video_name))

    random.shuffle(all_frames)
    split_index = int(len(all_frames) * train_ratio)

    # Load pre-trained YOLOv5 model
    model = YOLO('yolov5su.pt')

    for i, (frame_path, video_name) in enumerate(tqdm(all_frames, desc="Processing frames")):
        subset = "train" if i < split_index else "val"
        frame_name = os.path.basename(frame_path)
        new_frame_name = f"{video_name}_{frame_name}"
        
        new_frame_path = os.path.join(images_dir, subset, new_frame_name)
        shutil.copy(frame_path, new_frame_path)  # Use copy instead of move

        label_path = os.path.join(labels_dir, subset, new_frame_name.replace('.jpg', '.txt'))
        annotate_image(model, new_frame_path, label_path)

    # Remove temporary directories after processing
    for video_file in os.listdir(videos_dir):
        if video_file.endswith((".mp4", ".avi", ".mov")):
            video_name = os.path.splitext(video_file)[0]
            frames_dir = os.path.join(output_dir, f"temp_frames_{video_name}")
            shutil.rmtree(frames_dir)

    print(f"Dataset prepared and annotated in {output_dir}")

if __name__ == "__main__":
    videos_dir = "data/raw/videos"
    output_dir = "data/processed"
    prepare_dataset(videos_dir, output_dir)
