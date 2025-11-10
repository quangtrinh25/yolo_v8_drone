import os
import json
import cv2
import shutil
import yaml
from sklearn.model_selection import train_test_split
from collections import defaultdict

COMPETITION_TRAIN_PATH = r"D:/zalo_ai/train"

def create_competition_dataset():
    """Tạo competition dataset với YOLOv8"""
    
    annotations_path = os.path.join(COMPETITION_TRAIN_PATH, 'annotations', 'annotations.json')
    samples_dir = os.path.join(COMPETITION_TRAIN_PATH, 'samples')
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Phân tích distribution
    object_stats = defaultdict(list)
    for ann in annotations:
        object_type = ann['video_id'].split('_')[0]
        object_stats[object_type].append(ann['video_id'])
    
    print("Object Distribution:")
    for obj_type, videos in object_stats.items():
        print(f"  {obj_type}: {len(videos)} videos")
    
    # Tạo stratified split
    train_videos, val_videos = [], []
    for obj_type, videos in object_stats.items():
        if len(videos) == 1:
            train_videos.extend(videos)
        else:
            obj_train, obj_val = train_test_split(videos, test_size=0.2, random_state=42)
            train_videos.extend(obj_train)
            val_videos.extend(obj_val)
    
    print(f"Train: {len(train_videos)} videos, Val: {len(val_videos)} videos")
    
    # Tạo thư mục
    output_dir = 'competition_data'
    for split in ['train', 'val']:
        os.makedirs(f'{output_dir}/images/{split}', exist_ok=True)
        os.makedirs(f'{output_dir}/labels/{split}', exist_ok=True)
    
    # Xử lý từng video
    for ann in annotations:
        video_id = ann['video_id']
        video_path = os.path.join(samples_dir, video_id, 'drone_video.mp4')
        
        if video_id in train_videos:
            split = 'train'
        else:
            split = 'val'
        
        print(f"Processing {video_id} -> {split}")
        
        if not os.path.exists(video_path):
            continue
            
        # Trích xuất frames có annotations
        cap = cv2.VideoCapture(video_path)
        extracted_count = 0
        
        for bbox_group in ann['annotations']:
            for bbox_info in bbox_group['bboxes']:
                frame_num = bbox_info['frame']
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    # Lưu frame
                    frame_filename = f"{video_id}_frame_{frame_num:06d}.jpg"
                    frame_path = f'{output_dir}/images/{split}/{frame_filename}'
                    cv2.imwrite(frame_path, frame)
                    
                    # Tạo label
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = bbox_info['x1'], bbox_info['y1'], bbox_info['x2'], bbox_info['y2']
                    
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # Clamp coordinates
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.001, min(1.0, width))
                    height = max(0.001, min(1.0, height))
                    
                    label_filename = f"{video_id}_frame_{frame_num:06d}.txt"
                    label_path = f'{output_dir}/labels/{split}/{label_filename}'
                    
                    with open(label_path, 'w') as f:
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    extracted_count += 1
        
        cap.release()
        print(f"  Extracted {extracted_count} frames")
        
        # Thêm reference images
        ref_dir = os.path.join(samples_dir, video_id, 'object_images')
        if os.path.exists(ref_dir):
            for i, img_file in enumerate(os.listdir(ref_dir)):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    src_path = os.path.join(ref_dir, img_file)
                    dst_path = f'{output_dir}/images/{split}/{video_id}_ref_{i}.jpg'
                    shutil.copy2(src_path, dst_path)
                    
                    label_path = f'{output_dir}/labels/{split}/{video_id}_ref_{i}.txt'
                    with open(label_path, 'w') as f:
                        f.write("0 0.5 0.5 0.8 0.8\n")
    
    # Tạo data.yaml
    config = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['target_object']
    }
    
    with open(f'{output_dir}/data.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("Competition dataset created!")

if __name__ == "__main__":
    create_competition_dataset()