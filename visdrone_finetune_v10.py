import os
import yaml
import shutil
import cv2
from ultralytics import YOLO

def create_visdrone_dataset():
    """Tạo dataset từ VisDrone structure thực tế với xử lý lỗi tọa độ"""
    
    # Tạo thư mục YOLO format
    os.makedirs('visdrone_yolo/images/train', exist_ok=True)
    os.makedirs('visdrone_yolo/labels/train', exist_ok=True)
    os.makedirs('visdrone_yolo/images/val', exist_ok=True)
    os.makedirs('visdrone_yolo/labels/val', exist_ok=True)
    
    # VisDrone classes (10 classes)
    visdrone_classes = ['pedestrian', 'people', 'bicycle', 'car', 'van', 
                       'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    
    # Xử lý train và val sets
    for split in ['train', 'val']:
        if split == 'train':
            base_path = r"D:/zalo_ai/VisDrone2019-DET-train"
        else:
            base_path = r"D:/zalo_ai/VisDrone2019-DET-val"
            
        images_dir = os.path.join(base_path, 'images')
        labels_dir = os.path.join(base_path, 'annotations')
        
        print(f"Processing {split} set: {images_dir}")
        
        # Xử lý từng ảnh
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files[:200]:
            img_path = os.path.join(images_dir, img_file)
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            # Copy image
            shutil.copy2(img_path, f'visdrone_yolo/images/{split}/{img_file}')
            
            # Convert annotation nếu tồn tại
            if os.path.exists(label_path):
                convert_visdrone_annotation(label_path, f'visdrone_yolo/labels/{split}/{label_file}', img_path)
    
    # Tạo data.yaml
    config = {
        'path': os.path.abspath('visdrone_yolo'),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 10,
        'names': visdrone_classes
    }
    
    with open('visdrone_yolo/data.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("VisDrone dataset created!")

def convert_visdrone_annotation(src_label_path, dst_label_path, img_path):
    """Chuyển đổi VisDrone annotation sang YOLO format với xử lý lỗi tọa độ"""
    try:
        # Đọc kích thước ảnh thực tế
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read image {img_path}, using default size 1920x1080")
            img_width, img_height = 1920, 1080
        else:
            img_height, img_width = img.shape[:2]
        
        with open(src_label_path, 'r') as f:
            lines = f.readlines()
        
        yolo_lines = []
        valid_boxes = 0
        invalid_boxes = 0
        
        for line in lines:
            data = line.strip().split(',')
            if len(data) < 6:
                continue
                
            try:
                x_left, y_top, width, height = float(data[0]), float(data[1]), float(data[2]), float(data[3])
                class_id = int(data[5]) - 1
                
                if class_id < 0 or class_id > 9:
                    continue
                
                # Tính toán tọa độ YOLO
                x_center = (x_left + width / 2) / img_width
                y_center = (y_top + height / 2) / img_height
                w_norm = width / img_width
                h_norm = height / img_height
                
                # Kiểm tra và clamp tọa độ về [0,1]
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.001, min(1.0, w_norm))
                h_norm = max(0.001, min(1.0, h_norm))
                
                if w_norm <= 0.001 or h_norm <= 0.001:
                    invalid_boxes += 1
                    continue
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                valid_boxes += 1
                
            except (ValueError, IndexError):
                invalid_boxes += 1
                continue
        
        with open(dst_label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
            
        if invalid_boxes > 0:
            print(f"  {os.path.basename(src_label_path)}: {valid_boxes} valid, {invalid_boxes} invalid boxes")
            
    except Exception as e:
        print(f"Error converting {src_label_path}: {e}")

def main():
    print("Starting YOLOv8s VisDrone Fine-tuning...")
    
    # Tạo dataset (bỏ qua nếu đã có)
    if not os.path.exists('visdrone_yolo/data.yaml'):
        create_visdrone_dataset()
    else:
        print("Using existing VisDrone dataset...")
    
    # Load YOLOv8s model
    print("Loading YOLOv8s model...")
    model = YOLO('yolov8s.pt')
    print("YOLOv8s model loaded successfully!")
    
    # Fine-tune
    print("Starting fine-tuning...")
    results = model.train(
        data='visdrone_yolo/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        lr0=0.01,
        patience=10,
        device='cpu',
        workers=0,
        project='yolov8_visdrone',
        name='visdrone_finetune',
        augment=True,
        cache=False,
        save_period=5,
        single_cls=False,
        verbose=True
    )
    
    print("VisDrone fine-tuning completed!")

if __name__ == "__main__":
    main()