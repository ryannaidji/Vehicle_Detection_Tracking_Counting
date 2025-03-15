import cv2
import numpy as np
import os
from ultralytics import YOLO
import norfair
from norfair import Tracker, Detection

# Chargement du modèle YOLO
MODEL_PATH = os.path.join("models", "yolo11n.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found in {MODEL_PATH}")

model = YOLO(MODEL_PATH)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return "Error: Unable to open the video."
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30
    
    counting_line = frame_height // 2
    
    vehicle_counter = 0
    tracker = Tracker(distance_function="euclidean", distance_threshold=30)
    tracked_vehicles = set()
    car_class_id = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        results = model(frame)
        detections = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                centroid_x = (x1 + x2) // 2  
                centroid_y = (y1 + y2) // 2  
                class_id = int(box.cls[0])  
                label = model.names[class_id]  
                
                if car_class_id is None and "car" in model.names.values():
                    car_class_id = [k for k, v in model.names.items() if v == "car"][0]
                
                if class_id != car_class_id:
                    continue  
                
                detections.append(Detection(points=np.array([[centroid_x, centroid_y]])))
        
        tracked_objects = tracker.update(detections=detections)
        
        for obj in tracked_objects:
            centroid_x, centroid_y = obj.estimate[0]  
            obj_id = obj.id  
            
            if counting_line - 5 < centroid_y < counting_line + 5 and obj_id not in tracked_vehicles:
                vehicle_counter += 1
                tracked_vehicles.add(obj_id)  
    
    cap.release()
    
    return vehicle_counter
