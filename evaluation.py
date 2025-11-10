import json
import os

def calculate_st_iou(gt_annotations, pred_annotations):
    """Tính ST-IoU score"""
    total_st_iou = 0
    video_count = 0
    
    for gt_ann in gt_annotations:
        video_id = gt_ann['video_id']
        pred_ann = next((p for p in pred_annotations if p['video_id'] == video_id), None)
        if not pred_ann:
            continue
            
        gt_frames = {}
        for bbox_group in gt_ann['annotations']:
            for bbox in bbox_group['bboxes']:
                gt_frames[bbox['frame']] = bbox
        
        pred_frames = {}
        for det_group in pred_ann['detections']:
            for bbox in det_group['bboxes']:
                pred_frames[bbox['frame']] = bbox
        
        intersection_frames = set(gt_frames.keys()) & set(pred_frames.keys())
        union_frames = set(gt_frames.keys()) | set(pred_frames.keys())
        
        if not union_frames:
            continue
            
        total_iou = 0
        for frame in intersection_frames:
            gt_bbox = gt_frames[frame]
            pred_bbox = pred_frames[frame]
            total_iou += calculate_bbox_iou(gt_bbox, pred_bbox)
        
        video_st_iou = total_iou / len(union_frames)
        total_st_iou += video_st_iou
        video_count += 1
        
        print(f"Video {video_id}: ST-IoU = {video_st_iou:.4f}")
    
    final_score = total_st_iou / video_count if video_count > 0 else 0
    print(f"Final ST-IoU Score: {final_score:.4f}")
    return final_score

def calculate_bbox_iou(bbox1, bbox2):
    """Tính IoU cho 2 bounding boxes"""
    x1_i = max(bbox1['x1'], bbox2['x1'])
    y1_i = max(bbox1['y1'], bbox2['y1'])
    x2_i = min(bbox1['x2'], bbox2['x2'])
    y2_i = min(bbox1['y2'], bbox2['y2'])
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
    area2 = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

if __name__ == "__main__":
    # Đánh giá trên train set để test
    gt_path = r"D:\zalo_ai\train\annotations\annotations.json"
    pred_path = 'yolov8_submission.json'
    
    if not os.path.exists(gt_path):
        print(f"Ground truth not found: {gt_path}")
        exit(1)
    
    if not os.path.exists(pred_path):
        print(f"Predictions not found: {pred_path}")
        exit(1)
    
    with open(gt_path, 'r') as f:
        gt_annotations = json.load(f)
    
    with open(pred_path, 'r') as f:
        pred_annotations = json.load(f)
    
    score = calculate_st_iou(gt_annotations, pred_annotations)
    print(f"Overall ST-IoU: {score:.4f}")