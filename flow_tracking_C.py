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
        
        # Tracking zones and movements
        self.zone_tracks = {
            'A': set(),
            'B': set(),
            'C': set()
        }
        
        # Tracking zone transitions
        self.zone_transitions = {
            'A_to_B': set(),
            'A_to_C': set(),
            'B_to_A': set(),
            'B_to_C': set(),
            'C_to_A': set(),
            'C_to_B': set()
        }

        
        # Track previous zone location for each track
        self.track_previous_zone = {}
        self.frame_transitions = defaultdict(int)
        self.total_zone_transitions = defaultdict(int)

        
        self.encoder = FeatureExtractor()

    def detect_and_track(self, frame, zoneA, zoneB, zoneC):
        results = self.model(frame, conf=0.5)[0]
        self.frame_transitions.clear()

        
        # Get raw detections for visualization and tracking
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

        # Process tracks - get positions, update trails, track zone occupancy
        track_data = []
        current_zone_tracks = {'A': set(), 'B': set(), 'C': set()}
        
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
            
            # Determine zone occupancy
            current_zone = None
            if cv2.pointPolygonTest(zoneA, (float(center_x), float(center_y)), False) >= 0:
                current_zone = 'A'
                current_zone_tracks['A'].add(track_id)
            elif cv2.pointPolygonTest(zoneB, (float(center_x), float(center_y)), False) >= 0:
                current_zone = 'B'
                current_zone_tracks['B'].add(track_id)
            elif cv2.pointPolygonTest(zoneC, (float(center_x), float(center_y)), False) >= 0:
                current_zone = 'C'
                current_zone_tracks['C'].add(track_id)
            
            # Track zone transitions
            if current_zone:
                previous_zone = self.track_previous_zone.get(track_id)
                
                # Check and record zone transitions
                if previous_zone and previous_zone != current_zone:
                    transition_key = f'{previous_zone}_to_{current_zone}'
                    self.zone_transitions[transition_key].add(track_id)
                    self.frame_transitions[transition_key] += 1
                    self.total_zone_transitions[transition_key] += 1

                # Update previous zone
                self.track_previous_zone[track_id] = current_zone
            
            # Add to output data
            track_data.append({
                'id': track_id,
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'confidence': track.det_conf,
                'zone': current_zone
            })
        
        # Update zone tracks
        self.zone_tracks = current_zone_tracks
        
        return track_data, raw_detections, current_zone_tracks

def setup_counting_zones(width, height, zone_config=None):
    #For IntersectionC - Session 15th feb
    zoneA = np.array([[1615, 193], [1615, 327],[21, 382],[148, 245]], np.int32)
    zoneB = np.array([[1620, 192], [1623, 317],[3032, 175],[2755, 112]], np.int32)
    zoneC = np.array([[500, 880],[2965,800], [2962, 230], [550, 527]], np.int32) 
    return zoneA, zoneB, zoneC

def process_video(video, output_video=True):

    session="Session_02152024"
    output_dir = f'/home/schivilkar/dev/final_video_processing/{session}/IntersectionC_final/{video}'
    os.makedirs(output_dir, exist_ok=True)

    output_dir_csv = f'/home/schivilkar/dev/final_video_processing/{session}/IntersectionC_final/FinalFlows'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_csv, exist_ok=True)

    output_dir2 = f'/home/schivilkar/dev/processed_video/{session}/IntersectionC/{video}'
    csv_name = video + "full_pedestrian_flow.csv"
    video_path = os.path.join(output_dir2, video+"_CROPPED.MP4")

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frame count:", total_frames)
    
    zoneA, zoneB, zoneC = setup_counting_zones(frame_width, frame_height)
    
    tracker = PedestrianTracker()
    
    # Set up CSV for data logging
    csv_path = os.path.join(output_dir_csv, csv_name)
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            'Frame', 
            'Total_Detections',
            'Total_Tracked',
            'Zone_A_Count', 
            'Zone_B_Count', 
            'Zone_C_Count',
            'A_to_B', 
            'A_to_C', 
            'B_to_A', 
            'B_to_C',
            'C_to_A', 
            'C_to_B',
            'Total_A_to_B', 
            'Total_A_to_C', 
            'Total_B_to_A', 
            'Total_B_to_C',
            'Total_C_to_A', 
            'Total_C_to_B'
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
        
        # Detect and track
        tracks, raw_detections, current_zone_tracks = tracker.detect_and_track(frame, zoneA, zoneB, zoneC)
        
        if output_video:
            vis_frame = frame.copy()
            
            # Draw zones
            cv2.polylines(vis_frame, [zoneA], True, (255, 0, 0), 2)  # Zone A in blue
            cv2.polylines(vis_frame, [zoneB], True, (0, 0, 255), 2)  # Zone D in red
            cv2.polylines(vis_frame, [zoneC], True, (0, 255, 0), 2)  # Zone C in green

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

            for track_data in tracks:
                track_id = track_data['id']
                x1, y1, x2, y2 = track_data['bbox']
                
                # Define color based on which zone the person is in
                if track_id in current_zone_tracks['A']:
                    color = (255, 0, 255)
                elif track_id in current_zone_tracks['B']:
                    color = (255, 0, 0) 
                elif track_id in current_zone_tracks['C']:
                    color = (0, 0, 255)  
                else:
                    color = (0, 255, 0)
                
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

            y_offset = 30
            # Add zone occupancy info
            cv2.putText(
                vis_frame,
                f'Frame: {frame_count} | A: {len(current_zone_tracks["A"])} B: {len(current_zone_tracks["B"])} C: {len(current_zone_tracks["C"])}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            y_offset +=30
            
            transition_colors = {
                'A_to_B': (255, 165, 0),  # Orange
                'A_to_C': (255, 0, 255),  # Magenta
                'B_to_A': (0, 165, 255),  # Light Orange
                'B_to_C': (0, 255, 255), # Cyan
                'C_to_A': (165, 255, 0), # Lime
                'C_to_B': (255, 0, 165)  # Pink
            }

            # Define table parameters
            start_x, start_y = 10, 50  # Starting position of the table
            row_height = 30  # Height of each row
            col_width = 150  # Width of columns
            header_height = 40  # Height of the header row
            table_width = col_width * 2  # Two columns: Transition and Count
            table_height = header_height + len(tracker.frame_transitions) * row_height  # Total table height

            # Draw table background
            cv2.rectangle(vis_frame, (start_x, start_y), 
                        (start_x + table_width, start_y + table_height), 
                        (50, 50, 50), -1) 

            # Draw header row
            cv2.rectangle(vis_frame, (start_x, start_y), 
                        (start_x + table_width, start_y + header_height), 
                        (100, 100, 100), -1) 

            # Add header text
            cv2.putText(vis_frame, 'Transition', (start_x + 10, start_y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, 'Count', (start_x + col_width + 10, start_y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw table rows with transition data
            y_offset = start_y + header_height
            for transition, count in tracker.total_zone_transitions.items():
                #print(tracker.total_zone_transitions.items())
                color = transition_colors.get(transition, (255, 255, 255))

                # Draw row background
                cv2.rectangle(vis_frame, (start_x, y_offset), 
                            (start_x + table_width, y_offset + row_height), 
                            (80, 80, 80), -1) 

                # Draw text
                cv2.putText(vis_frame, transition, (start_x + 10, y_offset + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(vis_frame, str(count), (start_x + col_width + 10, y_offset + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                y_offset += row_height


            writer.write(vis_frame)
            
            # Save keyframes
            if frame_count % 300 == 0:  
                frame_path = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
                cv2.imwrite(frame_path, vis_frame)
        
        with open(csv_path, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                frame_count,
                len(raw_detections),
                len(tracks),
                len(current_zone_tracks['A']),
                len(current_zone_tracks['B']),
                len(current_zone_tracks['C']),
                tracker.frame_transitions['A_to_B'],
                tracker.frame_transitions['A_to_C'],
                tracker.frame_transitions['B_to_A'],
                tracker.frame_transitions['B_to_C'],
                tracker.frame_transitions['C_to_A'],
                tracker.frame_transitions['C_to_B'],
                tracker.total_zone_transitions['A_to_B'],
                tracker.total_zone_transitions['A_to_C'],
                tracker.total_zone_transitions['B_to_A'],
                tracker.total_zone_transitions['B_to_C'],
                tracker.total_zone_transitions['C_to_A'],
                tracker.total_zone_transitions['C_to_B']
            ])
        
    cap.release()
    if output_video:
        writer.release()
    

def main():
    #For intersectionC - "GH020006","GH030006","GH040006","GH050006",
    video_list = ["GH010006", "GH060006","GH070006","GH080006","GH090006"]
   
    total_start_time = time.time()
    for video in video_list:
        start_time = time.time()
        process_video(video)
        elapsed_time = time.time() - start_time        
        print(f'\nProcessing time for: {video} is {elapsed_time:.2f} seconds')
    
    total_elapsed_time = time.time() - total_start_time
    print(f'Total time for all videos processing: {total_elapsed_time} seconds')

if __name__ == "__main__":
    main()
