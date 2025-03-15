import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
import norfair
from norfair import Tracker, Detection

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("highway.mp4")

if not cap.isOpened():
    print("Error: Unable to open the video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
counting_line = frame_height // 2  

vehicle_counter = 0
tracker = Tracker(distance_function="euclidean", distance_threshold=30)
tracked_vehicles = set()

vehicle_paths = {}

car_class_id = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    tracked_objects = tracker.update(detections=detections)

    for obj in tracked_objects:
        centroid_x, centroid_y = obj.estimate[0]  
        obj_id = obj.id  

        if counting_line - 5 < centroid_y < counting_line + 5 and obj_id not in tracked_vehicles:
            vehicle_counter += 1
            tracked_vehicles.add(obj_id)  

        if obj_id not in vehicle_paths:
            vehicle_paths[obj_id] = []
        vehicle_paths[obj_id].append((int(centroid_x), int(centroid_y)))

        vehicle_paths[obj_id] = vehicle_paths[obj_id][-20:]

        for i in range(1, len(vehicle_paths[obj_id])):
            cv2.line(frame, vehicle_paths[obj_id][i - 1], vehicle_paths[obj_id][i], (0, 0, 255), 2)

        cv2.circle(frame, (int(centroid_x), int(centroid_y)), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"ID {obj_id}", (int(centroid_x), int(centroid_y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.line(frame, (0, counting_line), (frame.shape[1], counting_line), (255, 0, 0), 2)
    cv2.putText(frame, f"Vehicles Counted: {vehicle_counter}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Vehicle Detection with Tracking and Paths", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame({"Total Vehicles Counted": [vehicle_counter]})
df.to_csv("vehicle_count.csv", index=False)

print(f"Total cars counted: {vehicle_counter}")
