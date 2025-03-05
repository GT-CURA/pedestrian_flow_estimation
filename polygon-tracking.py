from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import os
import time
import csv
from collections import defaultdict

class PedestrianTracker:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.previous_positions = {}
        self.track_history = defaultdict(list)
        
    def detect_pedestrians(self, frame, conf_threshold=0.5):
        results = self.model(frame)
        detections_df = results[0].boxes.data.cpu().numpy()
        df = pd.DataFrame(detections_df, columns=["x1", "y1", "x2", "y2", "confidence", "class"])
        
        detections = df[(df["confidence"] > conf_threshold) & (df["class"] == 0)]  # class 0 is person
        return detections[["x1", "y1", "x2", "y2", "confidence"]].values

def setup_counting_zones(width, height, zone_config=None):
    if zone_config is None:
        # left_zone = np.array([[int(width*0.3), int(height*0.4)], 
        #                     [int(width*0.4), int(height*0.8)],
        #                     [int(width*0.45), int(height*0.8)],
        #                     [int(width*0.35), int(height*0.4)]], np.int32)
        
        # right_zone = np.array([[int(width*0.35), int(height*0.4)],
        #                       [int(width*0.45), int(height*0.8)],
        #                       [int(width*0.5), int(height*0.8)],
        #                       [int(width*0.4), int(height*0.4)]], np.int32)
        left_zone = np.array([[int(width*0.2), int(height*0.4)],  # Move left boundary further left
                          [int(width*0.4), int(height*0.8)],  
                          [int(width*0.5), int(height*0.8)],  # Extend right boundary further right
                          [int(width*0.3), int(height*0.4)]], np.int32)

    # Make the right zone horizontal and shift downward
        # right_zone = np.array([[int(width*0.35), int(height*0.65)],  # Shift zone downward
        #                    [int(width*0.55), int(height*0.65)],  # Make it wider
        #                    [int(width*0.55), int(height*0.75)],  # Keep height consistent
        #                    [int(width*0.35), int(height*0.75)]], np.int32)
        right_zone = np.array([[int(width*0.45), int(height*0.5)],  # Shift zone downward
                           [int(width*0.75), int(height*0.5)],  # Make it wider
                           [int(width*0.55), int(height*0.75)],  # Keep height consistent
                           [int(width*0.35), int(height*0.75)]], np.int32)
    else:
        left_zone = np.array(zone_config['left'], np.int32)
        right_zone = np.array(zone_config['right'], np.int32)
    
    return left_zone, right_zone

def process_video(video_path, output_dir, output_video=True):
    tracker = PedestrianTracker()
    capture = cv2.VideoCapture(video_path)
    
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    left_zone, right_zone = setup_counting_zones(frame_width, frame_height)
    
    # Initialize counters
    left_count = 0
    right_count = 0
    frame_number = 0
    
    if output_video:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'output.mp4')
        writer = cv2.VideoWriter(output_path, 
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, (frame_width, frame_height))
    
    csv_path = os.path.join(output_dir, 'pedestrian_counts.csv')
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['Frame', 'Incremental_Left', 'Incremental_Right', 
                           'Total_Left', 'Total_Right'])
    
    # Track pedestrians through zones
    while True:
        ret, frame = capture.read()
        if not ret:
            break
            
        frame_number += 1
        incremental_left = 0
        incremental_right = 0
        
        detections = tracker.detect_pedestrians(frame)
        
        for det in detections:
            x1, y1, x2, y2, conf = det
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            point = np.array([center_x, center_y])
            in_left = cv2.pointPolygonTest(left_zone, (float(center_x), float(center_y)), False) >= 0
            in_right = cv2.pointPolygonTest(right_zone, (float(center_x), float(center_y)), False) >= 0
            
            if in_left and not in_right:
                incremental_left += 1
                left_count += 1
            elif in_right and not in_left:
                incremental_right += 1
                right_count += 1
        
        if output_video:
            cv2.polylines(frame, [left_zone], True, (255, 0, 0), 2)
            cv2.polylines(frame, [right_zone], True, (0, 0, 255), 2)
            
            for det in detections:
                x1, y1, x2, y2, conf = det
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            cv2.putText(frame, f'Left: {left_count} Right: {right_count}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            writer.write(frame)
            
            if frame_number % 1 == 0:  
                frame_path = os.path.join(output_dir, f'frame_{frame_number}.jpg')
                cv2.imwrite(frame_path, frame)
        
        with open(csv_path, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([frame_number, incremental_left, incremental_right,
                               left_count, right_count])
        
        if frame_number % 100 == 0:
            print(f'Processed frame {frame_number}/{total_frames}')
    
    capture.release()
    if output_video:
        writer.release()
    
    return left_count, right_count

def main():
    video_directory = "/media/chan/backup_SSD2/ASPED.c/Session_02152024/IntersectionB/Video/gopro04/"
    output_base_dir = "output_frames3/"
    
    for root, _, files in os.walk(video_directory):
        for file in files:
            if file.endswith('GH010015.MP4'):
                video_path = os.path.join(root, file)
                video_output_dir = os.path.join(output_base_dir, os.path.splitext(file)[0])
                
                print(f'Processing {file}...')
                start_time = time.time()
                
                left_count, right_count = process_video(video_path, video_output_dir)
                
                elapsed_time = time.time() - start_time
                print(f'Finished processing {file}')
                print(f'Left count: {left_count}, Right count: {right_count}')
                print(f'Processing time: {elapsed_time:.2f} seconds')
                print('-' * 50)

if __name__ == "__main__":
    main()