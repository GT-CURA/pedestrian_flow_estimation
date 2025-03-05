import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
import time
import os
import csv
from collections import defaultdict
import torch
from torchvision import models
from torchvision import transforms



class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True).to(self.device)
        
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, frame, boxes):
        """Extracts features for detected objects."""
        features = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped = frame[y1:y2, x1:x2]  
            if cropped.size == 0:
                continue 
            img = self.transform(cropped).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.model(img)  # Extract features
            features.append(feature.squeeze().cpu().numpy())  
        return np.array(features)

class PedestrianTracker:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        
        self.tracker = DeepSort(
            max_age=6,
            n_init=2,
            max_iou_distance=0.9,
            max_cosine_distance=0.6,
            nn_budget=150
        )
        
        self.track_history = defaultdict(list)
        self.max_trail_length = 30
        
        self.tracks_in_left_zone = set()
        self.tracks_in_right_zone = set()
        self.counted_left = set()
        self.counted_right = set()
        self.encoder = FeatureExtractor()


    def detect_and_track(self, frame):
        results = self.model(frame, conf=0.5)[0]
        
        # Get raw detections for visualization
        raw_detections = []
        raw_detections1 = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) == 0: 
                raw_detections.append((float(x1), float(y1), float(x2), float(y2), float(score)))
                raw_detections1.append((float(x1), float(y1), float(x2), float(y2)))

        
        # Format detections for DeepSORT
        deepsort_detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) == 0: 
                deepsort_detections.append(([x1, y1, x2-x1, y2-y1], score, 'person'))
        
        features = self.encoder.extract_features(frame, raw_detections1)
        tracks = self.tracker.update_tracks(deepsort_detections, features, frame)

        
        # Process tracks - get positions, update trails
        track_data = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = int(track.track_id)
            ltrb = track.to_ltrb()
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Update trail for visualization
            self.track_history[track_id].append((center_x, center_y))
            if len(self.track_history[track_id]) > self.max_trail_length:
                self.track_history[track_id].pop(0)
            
            # Add to output data
            track_data.append({
                'id': track_id,
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'confidence': track.det_conf
            })
            
        return track_data, raw_detections

def setup_counting_zones(width, height, zone_config=None):
    if zone_config is None:
        # Define default zones if no config provided
        left_zone = np.array([[474,508], [799,540],[1535,857],[113,949]], np.int32)
        
        right_zone = np.array([[1076,613], [1363,703],[1815,498],[1725,476]], np.int32)
    else:
        left_zone = np.array(zone_config['left'], np.int32)
        right_zone = np.array(zone_config['right'], np.int32)
    
    return left_zone, right_zone

def process_video(video, output_video=True):
    session="Session_02152024"
    output_dir = f'/home/schivilkar/dev/video_processing/video/{session}/{video}'
    os.makedirs(output_dir, exist_ok=True)
    #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionB/Video/gopro04/{v}.MP4' -an -c:v copy '{out_dir}/{v}_MUTED.MP4'".format(s = session, v=video, out_dir=output_dir))
    os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionB/Video/gopro04/{v}.MP4' -an -c:v copy -t 30 '{out_dir}/{v}_MUTED_30s.MP4'".format(s=session, v=video, out_dir=output_dir))

    #video_directory = "/home/schivilkar/dev/video_labeling/video/Session_02152024/GH010015"
    video_name = video+"_MUTED_30s.MP4"
    csv_name = video+"pedestrian_flow.csv"
    video_path = os.path.join(output_dir, video_name)
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frame count:******************",total_frames)
    
    left_zone, right_zone = setup_counting_zones(frame_width, frame_height)
    
    tracker = PedestrianTracker()
    
    # Set up CSV for data logging
    csv_path = os.path.join(output_dir, csv_name)
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            'Frame', 'Raw_Detections', 'Tracked_Count', 
            'New_Left_Count', 'New_Right_Count', 
            'Total_Unique_Left', 'Total_Unique_Right',
            'Current_Left_Zone', 'Current_Right_Zone'
        ])
    
    # Set up video writer if needed
    if output_video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(output_dir, f'tracked_zones_{timestamp}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_filename,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
    
    # Initialize counters and tracking variables
    frame_count = 0
    start_time = time.time()
    
    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Get tracks with IDs and raw detections
        tracks, raw_detections = tracker.detect_and_track(frame)
        
        # Track zone occupancy for this frame
        current_left_ids = set()
        current_right_ids = set()
        new_left_count = 0
        new_right_count = 0
        
        # Process each tracked pedestrian
        for track_data in tracks:
            track_id = track_data['id']
            center_x, center_y = track_data['center']
            
            # Check zone containment
            in_left = cv2.pointPolygonTest(left_zone, (float(center_x), float(center_y)), False) >= 0
            in_right = cv2.pointPolygonTest(right_zone, (float(center_x), float(center_y)), False) >= 0
            
            # Update current zone occupants
            if in_left:
                current_left_ids.add(track_id)
                if track_id not in tracker.tracks_in_left_zone and track_id not in tracker.counted_left:
                    # New person entered left zone
                    new_left_count += 1
                    tracker.counted_left.add(track_id)
            
            if in_right:
                current_right_ids.add(track_id)
                if track_id not in tracker.tracks_in_right_zone and track_id not in tracker.counted_right:
                    # New person entered right zone
                    new_right_count += 1
                    tracker.counted_right.add(track_id)
        
        # Update zone tracking
        tracker.tracks_in_left_zone = current_left_ids
        tracker.tracks_in_right_zone = current_right_ids
        
        if output_video:
            vis_frame = frame.copy()
            
            # Draw zones
            cv2.polylines(vis_frame, [left_zone], True, (255, 0, 0), 2)  # Left zone in blue
            cv2.polylines(vis_frame, [right_zone], True, (0, 0, 255), 2)  # Right zone in red
            
            # # FIRST: Draw ALL raw detections
            for det in raw_detections:
                x1, y1, x2, y2, conf = det
                # Draw all raw detections in yellow
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
                cv2.putText(
                    vis_frame,
                    f"{conf:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )
            
            # SECOND: Draw tracks and trails
            for track_data in tracks:
                track_id = track_data['id']
                x1, y1, x2, y2 = track_data['bbox']
                
                # Define color based on which zone the person is in
                if track_id in current_left_ids and track_id in current_right_ids:
                    color = (255, 0, 255)  # Purple for overlap
                elif track_id in current_left_ids:
                    color = (255, 0, 0)    # Blue for left zone
                elif track_id in current_right_ids:
                    color = (0, 0, 255)    # Red for right zone
                else:
                    color = (0, 255, 0)    # Green for no zone
                
                # Draw bounding box and ID
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    vis_frame, 
                    f"ID: {track_id}", 
                    (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2
                )
                
                # Draw trail
                points = np.array(tracker.track_history[track_id], np.int32)
                if len(points) > 1:
                    points = points.reshape((-1, 1, 2))
                    cv2.polylines(vis_frame, [points], False, color, 2)
            
            # Add counters and frame number
            cv2.putText(
                vis_frame,
                f'Frame: {frame_count} | Raw Detections: {len(raw_detections)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.putText(
                vis_frame,
                f'Tracked: {len(tracks)} | Left: {len(tracker.counted_left)} | Right: {len(tracker.counted_right)}',
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            writer.write(vis_frame)
            
            # Save keyframes
            if frame_count % 30 == 0:  
                frame_path = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
                cv2.imwrite(frame_path, vis_frame)
        
        # Log data to CSV
        with open(csv_path, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                frame_count,
                len(raw_detections),
                len(tracks),
                new_left_count,  #new how many left
                new_right_count,
                len(tracker.counted_left),  #total how many left
                len(tracker.counted_right),
                len(current_left_ids), #currently how many left
                len(current_right_ids) #currently how many right
            ])
        
    cap.release()
    if output_video:
        writer.release()
    
    return len(tracker.counted_left), len(tracker.counted_right)

def main():
    #video_list = ["GH010015", "GH020015","GH030015","GH040015","GH050015","GH060015", "GH070015","GH080015","GH090015","GH100015"]
    video_list = ["GH010015", "GH020015"]
    #GH010006.MP4  GH020006.MP4  GH030006.MP4  GH040006.MP4  GH050006.MP4  GH060006.MP4  GH070006.MP4  GH080006.MP4  GH090006.MP4
    #video_list = ["GH010006"]


    total_start_time = time.time()
    for video in video_list:
        start_time = time.time()
        left_count, right_count = process_video(video)
        elapsed_time = time.time() - start_time
        print(f'Unique pedestrians - Left zone: {left_count}, Right zone: {right_count}')
        print(f'Processing time for: {video} is {elapsed_time:.2f} seconds')
    total_elapsed_time = time.time() - total_start_time
    print(f'Total time for all videos processing: {total_elapsed_time} seconds')
if __name__ == "__main__":
    main()