import os
import json
import cv2
from tqdm import tqdm
from ultralytics import YOLO

COMPETITION_PUBLIC_TEST_PATH = r"D:/zalo_ai/public_test"

def inference_video(video_path, model, confidence_threshold=0.5):
    """Inference trên video với YOLOv8"""
    cap = cv2.VideoCapture(video_path)
    detections = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Sample mỗi 5 frames để tăng tốc
        if frame_count % 5 != 0:
            continue
        
        results = model(frame, conf=confidence_threshold, verbose=False)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    detections.append({
                        'frame': frame_count,
                        'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                        'confidence': float(conf)
                    })
    
    cap.release()
    return detections

def group_detections(detections, max_gap=5):
    """Nhóm detections liên tiếp"""
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda x: x['frame'])
    groups = []
    current_group = [detections[0]]
    
    for i in range(1, len(detections)):
        if detections[i]['frame'] - current_group[-1]['frame'] <= max_gap:
            current_group.append(detections[i])
        else:
            groups.append(current_group)
            current_group = [detections[i]]
    
    if current_group:
        groups.append(current_group)
    
    formatted_groups = []
    for group in groups:
        bboxes = []
        for det in group:
            bboxes.append({
                'frame': det['frame'],
                'x1': det['x1'], 'y1': det['y1'],
                'x2': det['x2'], 'y2': det['y2']
            })
        formatted_groups.append({'bboxes': bboxes})
    
    return formatted_groups

def create_submission():
    """Tạo submission file với YOLOv8"""
    
    # Load model
    try:
        model = YOLO('yolov8_competition/weights/best.pt')
        print("Loaded competition-trained YOLOv8 model")
    except:
        try:
            model = YOLO('yolov8_visdrone/weights/best.pt')
            print("Loaded VisDrone-trained YOLOv8 model")
        except:
            model = YOLO('yolov8s.pt')
            print("Using base YOLOv8s model")
    
    submission_data = []
    samples_dir = os.path.join(COMPETITION_PUBLIC_TEST_PATH, 'samples')
    test_videos = [d for d in os.listdir(samples_dir) if os.path.isdir(os.path.join(samples_dir, d))]
    
    for video_id in tqdm(test_videos):
        video_path = os.path.join(samples_dir, video_id, 'drone_video.mp4')
        
        if os.path.exists(video_path):
            detections = inference_video(video_path, model, confidence_threshold=0.5)
            detection_groups = group_detections(detections)
        else:
            detection_groups = []
        
        submission_data.append({
            'video_id': video_id,
            'detections': detection_groups
        })
    
    # Save submission
    with open('yolov8_submission.json', 'w') as f:
        json.dump(submission_data, f, indent=2)
    
    print("YOLOv8 submission created: yolov8_submission.json")

if __name__ == "__main__":
    create_submission()