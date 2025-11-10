import os
from ultralytics import YOLO

def train_competition_model():
    """Training competition với YOLOv8"""
    
    print("Starting YOLOv8 Competition Training...")
    
    # Load model đã fine-tune VisDrone
    try:
        model = YOLO('yolov8_visdrone/weights/best.pt')
        print("Loaded fine-tuned YOLOv8 model")
    except:
        model = YOLO('yolov8s.pt')
        print("Using base YOLOv8s model")
    
    # Kiểm tra dataset competition
    if not os.path.exists('competition_data/data.yaml'):
        print("Competition dataset not found!")
        print("Please run: python prepare_competition_v8.py")
        return
    
    # Training
    print("Starting competition training...")
    results = model.train(
        data='competition_data/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        lr0=0.001,
        patience=20,
        device='cpu',
        workers=0,
        project='yolov8_competition',
        name='competition_train',
        augment=True,
        degrees=15,
        translate=0.2,
        scale=0.5,
        flipud=0.3
    )
    
    print("Competition training completed!")

if __name__ == "__main__":
    train_competition_model()