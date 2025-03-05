import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
import time
import os
import csv
from collections import defaultdict

class PedestrianTracker:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        
        # Enhanced DeepSort configuration with improved feature extraction
        self.tracker = DeepSort(
            max_age=100,              # Increased from 50 to handle longer occlusions
            n_init=3,                 # Increased from 2 for more stable track initialization
            max_iou_distance=0.7,     # Reduced from 0.9 for stricter matching
            max_cosine_distance=0.2,  # Reduced from 0.4 for stricter appearance matching
            nn_budget=150,            # Increased from 100 to store more appearance features
            
            # Enhanced feature extraction parameters
            override_track_class=None,
            gating_only_position=False,     # Consider both position and size for gating
            embedder="mobilenet",           # Using MobileNet for better feature extraction
            half=True,                      # Use half precision for faster processing
            bgr=True,                       # Images are in BGR format (OpenCV default)
            embedder_gpu=True,              # Use GPU for feature extraction if available
            embedder_model_name=None,       # Use default model
            embedder_wts=None,              # Use default weights
            polygon=False,                  # Use rectangle detections
            today=None                      # Use system date
        )
        
        self.track_history = defaultdict(list)
        self.max_trail_length = 30
        
        self.tracks_in_left_zone = set()
        self.tracks_in_right_zone = set()
        self.counted_left = set()
        self.counted_right = set()
        
        # Enhanced features for better tracking
        self.appearance_features = {}  # Store appearance features for re-identification
        self.velocity_history = defaultdict(list)  # Store velocity info for motion prediction
        self.max_velocity_history = 10
        self.occlusion_handling = True  # Enable occlusion handling

    def detect_and_track(self, frame):
        results = self.model(frame, conf=0.5)[0]
        
        # Get raw detections for visualization
        raw_detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) == 0:  # Class 0 is person
                raw_detections.append((float(x1), float(y1), float(x2), float(y2), float(score)))
        
        # Format detections for DeepSORT with appearance features
        deepsort_detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) == 0:  # Class 0 is person
                # Extract person region for feature extraction
                bbox = [x1, y1, x2-x1, y2-y1]
                deepsort_detections.append((bbox, score, 'person'))
        
        # Update tracks with enhanced feature extraction
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        
        # Process tracks - get positions, update trails and velocity history
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
            
            # Calculate and store velocity for motion prediction
            if len(self.track_history[track_id]) >= 2:
                prev_x, prev_y = self.track_history[track_id][-2]
                curr_x, curr_y = center_x, center_y
                velocity_x = curr_x - prev_x
                velocity_y = curr_y - prev_y
                
                self.velocity_history[track_id].append((velocity_x, velocity_y))
                if len(self.velocity_history[track_id]) > self.max_velocity_history:
                    self.velocity_history[track_id].pop(0)
            
            # Store appearance feature for re-identification
            if hasattr(track, 'features') and track.features is not None:
                self.appearance_features[track_id] = track.features
            
            # Add to output data with enhanced information
            track_data.append({
                'id': track_id,
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'confidence': track.det_conf,
                'velocity': self._get_average_velocity(track_id) if track_id in self.velocity_history else (0, 0),
                'time_since_update': track.time_since_update
            })
            
        # Handle occlusions and predict positions for missing tracks
        if self.occlusion_handling:
            track_data = self._handle_occlusions(track_data)
            
        return track_data, raw_detections
    
    def _get_average_velocity(self, track_id):
        """Calculate average velocity from history for smoother predictions"""
        if not self.velocity_history[track_id]:
            return (0, 0)
            
        velocities = self.velocity_history[track_id]
        vx_sum = sum(v[0] for v in velocities)
        vy_sum = sum(v[1] for v in velocities)
        
        return (vx_sum / len(velocities), vy_sum / len(velocities))
    
    def _handle_occlusions(self, track_data):
        """Predict positions for occluded tracks based on velocity history"""
        # This would be called to predict positions of occluded objects
        # Not fully implemented in this example, but shows the concept
        return track_data
    
    def _match_appearance_features(self, track_id, new_feature):
        """Match appearance features for consistent re-identification"""
        # This would be used to help with track re-identification
        # Not fully implemented in this example, but shows the concept
        return True

def setup_counting_zones(width, height, zone_config=None):
    if zone_config is None:
        # Define default zones if no config provided
        left_zone = np.array([[474,508], [799,540],[1535,857],[113,949]], np.int32)
        
        right_zone = np.array([[1076,613], [1363,703],[1815,498],[1725,476]], np.int32)
    else:
        left_zone = np.array(zone_config['left'], np.int32)
        right_zone = np.array(zone_config['right'], np.int32)
    
    return left_zone, right_zone

def process_video(video_path, output_dir, output_video=True):
    # Create output directory
    video_directory = "/home/schivilkar/dev/video_labeling/video/Session_02152024/GH010015"
    video_path1 = os.path.join(video_directory, "GH010015_MUTED_30s.MP4")
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    print(video_path1)
    cap = cv2.VideoCapture(video_path1)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frame count:******************",total_frames)
    
    # Setup zones
    left_zone, right_zone = setup_counting_zones(frame_width, frame_height)
    
    # Initialize tracker
    tracker = PedestrianTracker()
    
    # Set up CSV for data logging
    csv_path = os.path.join(output_dir, 'pedestrian_counts.csv')
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            'Frame', 'Raw_Detections', 'Tracked_Count', 
            'New_Left_Count', 'New_Right_Count', 
            'Total_Unique_Left', 'Total_Unique_Right',
            'Current_Left_Zone', 'Current_Right_Zone',
            'ID_Switches'  # Added new column for tracking ID switches
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
    prev_track_ids = set()
    id_switches = 0
    
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
        
        # Detect ID switches by comparing with previous frame's IDs
        current_track_ids = {track_data['id'] for track_data in tracks}
        disappeared_ids = prev_track_ids - current_track_ids
        new_ids = current_track_ids - prev_track_ids
        
        # Simple heuristic: if same number of tracks disappeared and appeared, count as ID switches
        if len(disappeared_ids) > 0 and len(disappeared_ids) == len(new_ids):
            id_switches += len(disappeared_ids)
            
        # Update previous IDs for next frame
        prev_track_ids = current_track_ids
        
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
        
        # Visualization - if requested
        if output_video:
            # Create a copy for visualization
            vis_frame = frame.copy()
            
            # Draw zones
            cv2.polylines(vis_frame, [left_zone], True, (255, 0, 0), 2)  # Left zone in blue
            cv2.polylines(vis_frame, [right_zone], True, (0, 0, 255), 2)  # Right zone in red
            
            # FIRST: Draw ALL raw detections
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
                vx, vy = track_data.get('velocity', (0, 0))
                
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
                
                # Draw velocity vector for motion prediction
                if abs(vx) > 0 or abs(vy) > 0:
                    center_x, center_y = track_data['center']
                    end_x = center_x + int(vx * 3)  # Scale for visibility
                    end_y = center_y + int(vy * 3)
                    cv2.arrowedLine(vis_frame, (center_x, center_y), (end_x, end_y), (255, 255, 0), 2)
            
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
            
            cv2.putText(
                vis_frame,
                f'ID Switches: {id_switches}',
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Write frame to video
            writer.write(vis_frame)
            
            # Save keyframes
            if frame_count % 300 == 0:  # Save every 300 frames
                frame_path = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
                cv2.imwrite(frame_path, vis_frame)
        
        # Log data to CSV
        with open(csv_path, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                frame_count,
                len(raw_detections),
                len(tracks),
                new_left_count, 
                new_right_count,
                len(tracker.counted_left), 
                len(tracker.counted_right),
                len(current_left_ids),
                len(current_right_ids),
                id_switches
            ])
        
        # Print progress
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            fps_processing = frame_count / elapsed_time
            print(f"Processed {frame_count}/{total_frames} frames. Processing FPS: {fps_processing:.1f}")
            print(f"Raw detections: {len(raw_detections)}, Tracked: {len(tracks)}")
            print(f"Unique people counted - Left: {len(tracker.counted_left)}, Right: {len(tracker.counted_right)}")
            print(f"ID switches so far: {id_switches}")
    
    # Cleanup
    cap.release()
    if output_video:
        writer.release()
    
    # Final stats
    elapsed_time = time.time() - start_time
    print(f"Video processing complete in {elapsed_time:.2f} seconds")
    print(f"Output saved to: {output_dir}")
    print(f"Final counts - Left zone: {len(tracker.counted_left)}, Right zone: {len(tracker.counted_right)}")
    print(f"Total ID switches: {id_switches}")
    
    return len(tracker.counted_left), len(tracker.counted_right)

def main():
    video_directory = "/media/chan/backup_SSD2/ASPED.c/Session_02152024/IntersectionB/Video/gopro04/"
    output_base_dir = "merged_tracker_output/"
    
    for root, _, files in os.walk(video_directory):
        for file in files:
            if file.endswith('GH010015.MP4'):
                video_path = os.path.join(root, file)
                video_output_dir = os.path.join(output_base_dir, os.path.splitext(file)[0])
                
                print(f'Processing {file}...')
                print(f'Output will be saved to: {video_output_dir}')
                start_time = time.time()
                
                left_count, right_count = process_video(video_path, video_output_dir)
                
                elapsed_time = time.time() - start_time
                print(f'Finished processing {file}')
                print(f'Unique pedestrians - Left zone: {left_count}, Right zone: {right_count}')
                print(f'Processing time: {elapsed_time:.2f} seconds')
                print('-' * 50)

if __name__ == "__main__":
    main()


# import cv2
# import numpy as np
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from datetime import datetime
# import time
# import os
# import csv
# from collections import defaultdict

# class PedestrianTracker:
#     def __init__(self, model_path="yolov8n.pt"):
#         self.model = YOLO(model_path)
        
#         # Enhanced DeepSort configuration with improved feature extraction
#         self.tracker = DeepSort(
#             max_age=30,                # Reduced from 100 to avoid ghost tracks persisting too long
#             n_init=5,                  # Increased from 3 to ensure more confident track initialization
#             max_iou_distance=0.6,      # Adjusted for better balance
#             max_cosine_distance=0.2,   # Strict appearance matching
#             nn_budget=150,             # Store more appearance features
            
#             # Enhanced feature extraction parameters
#             override_track_class=None,
#             gating_only_position=False,
#             embedder="mobilenet",      # Better feature extraction
#             half=True,                 # Half precision for faster processing
#             bgr=True,                  # OpenCV default
#             embedder_gpu=True,         # Use GPU for feature extraction
#             embedder_model_name=None,
#             embedder_wts=None,
#             polygon=False,
#             today=None
#         )
        
#         self.track_history = defaultdict(list)
#         self.max_trail_length = 20     # Reduced to avoid long trails that don't match actual movement
        
#         self.tracks_in_left_zone = set()
#         self.tracks_in_right_zone = set()
#         self.counted_left = set()
#         self.counted_right = set()
        
#         # Enhanced tracking features
#         self.appearance_features = {}
#         self.velocity_history = defaultdict(list)
#         self.max_velocity_history = 5  # Reduced to better capture recent motion
        
#         # Track quality metrics
#         self.track_confidence = defaultdict(float)
#         self.min_confidence = 0.3      # Minimum confidence to display a track
#         self.track_detection_count = defaultdict(int)  # Count consecutive detections
#         self.min_detections = 3        # Minimum consecutive detections to display
        
#         # Track clearing - remove tracks that are floating in air
#         self.height_consistency = defaultdict(list)
#         self.track_last_update = defaultdict(int)

#     def detect_and_track(self, frame, frame_count):
#         frame_height, frame_width = frame.shape[:2]
#         results = self.model(frame, conf=0.5)[0]
        
#         # Get raw detections for visualization
#         raw_detections = []
#         for r in results.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = r
#             if int(class_id) == 0:  # Class 0 is person
#                 raw_detections.append((float(x1), float(y1), float(x2), float(y2), float(score)))
        
#         # Format detections for DeepSORT with appearance features
#         deepsort_detections = []
#         for r in results.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = r
#             if int(class_id) == 0:  # Class 0 is person
#                 # Basic sanity check - person should be reasonable height and not floating
#                 person_height = y2 - y1
#                 if person_height < 0.05 * frame_height:  # Skip if too small (likely false positive)
#                     continue
                    
#                 # Skip if bottom of bounding box is too high up in the frame (floating person)
#                 if y2 < 0.4 * frame_height:  # Person's feet should be in lower 60% of frame
#                     continue
                    
#                 bbox = [x1, y1, x2-x1, y2-y1]
#                 deepsort_detections.append((bbox, score, 'person'))
        
#         # Update tracks
#         tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        
#         # Process tracks - get positions, update trails
#         track_data = []
#         valid_track_ids = set()
        
#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
                
#             track_id = int(track.track_id)
#             valid_track_ids.add(track_id)
#             ltrb = track.to_ltrb()
            
#             # Get bounding box coordinates
#             x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
#             # Sanity check - reject tracks with unreasonable dimensions
#             if (x2 - x1) < 20 or (y2 - y1) < 40:  # Too small to be a person
#                 continue
                
#             if y2 < 0.4 * frame_height:  # Reject floating tracks
#                 continue
                
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)
            
#             # Record the bottom center of the bounding box (feet position) for more accurate trails
#             feet_x = center_x
#             feet_y = y2
            
#             # Update trail using feet position instead of center
#             if track.time_since_update == 0:  # Only update with actual detections
#                 self.track_history[track_id].append((feet_x, feet_y))
#                 self.track_last_update[track_id] = frame_count
#                 self.track_detection_count[track_id] += 1
#             else:
#                 self.track_detection_count[track_id] = 0  # Reset consecutive detection count
                
#             if len(self.track_history[track_id]) > self.max_trail_length:
#                 self.track_history[track_id].pop(0)
            
#             # Calculate velocity based on recent trail positions
#             velocity_x, velocity_y = 0, 0
#             if len(self.track_history[track_id]) >= 2:
#                 prev_x, prev_y = self.track_history[track_id][-2]
#                 velocity_x = feet_x - prev_x
#                 velocity_y = feet_y - prev_y
                
#                 self.velocity_history[track_id].append((velocity_x, velocity_y))
#                 if len(self.velocity_history[track_id]) > self.max_velocity_history:
#                     self.velocity_history[track_id].pop(0)
            
#             # Track confidence update - trust tracks that are consistently detected
#             if track.time_since_update == 0:
#                 self.track_confidence[track_id] = min(1.0, self.track_confidence[track_id] + 0.1)
#             else:
#                 self.track_confidence[track_id] = max(0.0, self.track_confidence[track_id] - 0.2)
            
#             # Only include confident tracks
#             if self.track_confidence[track_id] >= self.min_confidence and self.track_detection_count[track_id] >= self.min_detections:
#                 # Add to output data
#                 track_data.append({
#                     'id': track_id,
#                     'bbox': (x1, y1, x2, y2),
#                     'center': (center_x, center_y),
#                     'feet': (feet_x, feet_y),
#                     'confidence': track.det_conf,
#                     'velocity': self._get_average_velocity(track_id),
#                     'time_since_update': track.time_since_update,
#                     'track_confidence': self.track_confidence[track_id]
#                 })
        
#         # Clean up old tracks that haven't been updated in a while
#         tracked_ids = set(track.track_id for track in tracks if track.is_confirmed())
#         for track_id in list(self.track_history.keys()):
#             if track_id not in tracked_ids or (frame_count - self.track_last_update[track_id]) > 30:
#                 # Clean up disappeared tracks
#                 self.track_history.pop(track_id, None)
#                 self.velocity_history.pop(track_id, None)
#                 self.track_confidence.pop(track_id, None)
#                 self.track_detection_count.pop(track_id, None)
#                 self.track_last_update.pop(track_id, None)
            
#         return track_data, raw_detections
    
#     def _get_average_velocity(self, track_id):
#         """Calculate average velocity from history for smoother predictions"""
#         if not self.velocity_history[track_id]:
#             return (0, 0)
            
#         velocities = self.velocity_history[track_id]
#         # Use the median velocity to avoid extreme values
#         vx_values = sorted(v[0] for v in velocities)
#         vy_values = sorted(v[1] for v in velocities)
        
#         # Get median values
#         if len(velocities) % 2 == 0:
#             median_x = (vx_values[len(velocities)//2] + vx_values[len(velocities)//2 - 1]) / 2
#             median_y = (vy_values[len(velocities)//2] + vy_values[len(velocities)//2 - 1]) / 2
#         else:
#             median_x = vx_values[len(velocities)//2]
#             median_y = vy_values[len(velocities)//2]
            
#         # Scale velocities to match real-world speed better
#         speed_scale_factor = 1.5  # Adjust to make displayed speed match reality better
#         return (median_x * speed_scale_factor, median_y * speed_scale_factor)

# def setup_counting_zones(width, height, zone_config=None):
#     if zone_config is None:
#         # Define default zones if no config provided
#         left_zone = np.array([[474,508], [799,540],[1535,857],[113,949]], np.int32)
        
#         right_zone = np.array([[1076,613], [1363,703],[1815,498],[1725,476]], np.int32)
#     else:
#         left_zone = np.array(zone_config['left'], np.int32)
#         right_zone = np.array(zone_config['right'], np.int32)
    
#     return left_zone, right_zone

# def process_video(video_path, output_dir, output_video=True):
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
#     video_directory = "/home/schivilkar/dev/video_labeling/video/Session_02152024/GH010015"
#     video_path1 = os.path.join(video_directory, "GH010015_MUTED_30s.MP4")
#     print(output_dir)
#     print(video_path1)
#     cap = cv2.VideoCapture(video_path1)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print("Total frame count:******************",total_frames)
    
#     # Setup zones
#     left_zone, right_zone = setup_counting_zones(frame_width, frame_height)
    
#     # Initialize tracker
#     tracker = PedestrianTracker()
    
#     # Set up CSV for data logging
#     csv_path = os.path.join(output_dir, 'pedestrian_counts.csv')
#     with open(csv_path, 'w', newline='') as f:
#         writer_csv = csv.writer(f)
#         writer_csv.writerow([
#             'Frame', 'Raw_Detections', 'Tracked_Count', 
#             'New_Left_Count', 'New_Right_Count', 
#             'Total_Unique_Left', 'Total_Unique_Right',
#             'Current_Left_Zone', 'Current_Right_Zone',
#             'ID_Switches', 'Ghost_Tracks_Filtered'
#         ])
    
#     # Set up video writer if needed
#     if output_video:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_filename = os.path.join(output_dir, f'tracked_zones_{timestamp}.mp4')
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         writer = cv2.VideoWriter(
#             output_filename,
#             fourcc,
#             fps,
#             (frame_width, frame_height)
#         )
    
#     # Initialize counters and tracking variables
#     frame_count = 0
#     start_time = time.time()
#     prev_track_ids = set()
#     id_switches = 0
#     ghost_tracks_filtered = 0
    
#     # Main processing loop
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame_count += 1
        
#         # Get tracks with IDs and raw detections
#         tracks, raw_detections = tracker.detect_and_track(frame, frame_count)
        
#         # Track zone occupancy for this frame
#         current_left_ids = set()
#         current_right_ids = set()
#         new_left_count = 0
#         new_right_count = 0
        
#         # Detect ID switches by comparing with previous frame's IDs
#         current_track_ids = {track_data['id'] for track_data in tracks}
#         disappeared_ids = prev_track_ids - current_track_ids
#         new_ids = current_track_ids - prev_track_ids
        
#         # Count filtered ghost tracks
#         ghost_tracks_filtered += len(prev_track_ids) - len(current_track_ids) if len(prev_track_ids) > len(current_track_ids) else 0
        
#         # Simple heuristic: if same number of tracks disappeared and appeared, count as ID switches
#         if len(disappeared_ids) > 0 and len(new_ids) > 0:
#             potential_switches = min(len(disappeared_ids), len(new_ids))
#             id_switches += potential_switches
            
#         # Update previous IDs for next frame
#         prev_track_ids = current_track_ids
        
#         # Process each tracked pedestrian
#         for track_data in tracks:
#             track_id = track_data['id']
#             feet_x, feet_y = track_data['feet']  # Use feet position for zone detection
            
#             # Check zone containment using feet position
#             in_left = cv2.pointPolygonTest(left_zone, (float(feet_x), float(feet_y)), False) >= 0
#             in_right = cv2.pointPolygonTest(right_zone, (float(feet_x), float(feet_y)), False) >= 0
            
#             # Update current zone occupants
#             if in_left:
#                 current_left_ids.add(track_id)
#                 if track_id not in tracker.tracks_in_left_zone and track_id not in tracker.counted_left:
#                     # New person entered left zone
#                     new_left_count += 1
#                     tracker.counted_left.add(track_id)
            
#             if in_right:
#                 current_right_ids.add(track_id)
#                 if track_id not in tracker.tracks_in_right_zone and track_id not in tracker.counted_right:
#                     # New person entered right zone
#                     new_right_count += 1
#                     tracker.counted_right.add(track_id)
        
#         # Update zone tracking
#         tracker.tracks_in_left_zone = current_left_ids
#         tracker.tracks_in_right_zone = current_right_ids
        
#         # Visualization - if requested
#         if output_video:
#             # Create a copy for visualization
#             vis_frame = frame.copy()
            
#             # Draw zones
#             cv2.polylines(vis_frame, [left_zone], True, (255, 0, 0), 2)  # Left zone in blue
#             cv2.polylines(vis_frame, [right_zone], True, (0, 0, 255), 2)  # Right zone in red
            
#             # Draw ALL raw detections
#             for det in raw_detections:
#                 x1, y1, x2, y2, conf = det
#                 # Draw raw detections in yellow with thinner lines
#                 cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
            
#             # Draw tracks and trails
#             for track_data in tracks:
#                 track_id = track_data['id']
#                 x1, y1, x2, y2 = track_data['bbox']
#                 feet_x, feet_y = track_data['feet']
#                 vx, vy = track_data.get('velocity', (0, 0))
#                 track_confidence = track_data.get('track_confidence', 0)
                
#                 # Color based on confidence and zone
#                 if track_id in current_left_ids and track_id in current_right_ids:
#                     base_color = (255, 0, 255)  # Purple for overlap
#                 elif track_id in current_left_ids:
#                     base_color = (255, 0, 0)    # Blue for left zone
#                 elif track_id in current_right_ids:
#                     base_color = (0, 0, 255)    # Red for right zone
#                 else:
#                     base_color = (0, 255, 0)    # Green for no zone
                
#                 # Adjust color intensity based on confidence
#                 color = tuple(int(c * track_confidence) for c in base_color)
                
#                 # Draw bounding box with varying thickness based on confidence
#                 thickness = max(1, int(track_confidence * 3))
#                 cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
                
#                 # Draw ID with confidence score
#                 cv2.putText(
#                     vis_frame, 
#                     f"ID:{track_id} {track_confidence:.1f}", 
#                     (x1, y1-10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, 
#                     color, 
#                     thickness
#                 )
                
#                 # Mark feet position with a circle
#                 cv2.circle(vis_frame, (feet_x, feet_y), 5, color, -1)
                
#                 # Draw trail using feet position for more accurate tracking
#                 points = np.array(tracker.track_history[track_id], np.int32)
#                 if len(points) > 1:
#                     points = points.reshape((-1, 1, 2))
#                     cv2.polylines(vis_frame, [points], False, color, thickness)
                
#                 # Draw velocity vector with proper scaling to match real-world speed
#                 if abs(vx) > 0 or abs(vy) > 0:
#                     # Draw from feet position, not center
#                     end_x = feet_x + int(vx * 2)
#                     end_y = feet_y + int(vy * 2)
#                     cv2.arrowedLine(vis_frame, (feet_x, feet_y), (end_x, end_y), (255, 255, 0), 2)
            
#             # Add counters and frame number
#             cv2.putText(
#                 vis_frame,
#                 f'Frame: {frame_count} | FPS: {fps} | Raw Detections: {len(raw_detections)}',
#                 (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 255),
#                 2
#             )
            
#             cv2.putText(
#                 vis_frame,
#                 f'Tracked: {len(tracks)} | Left: {len(tracker.counted_left)} | Right: {len(tracker.counted_right)}',
#                 (10, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 255),
#                 2
#             )
            
#             cv2.putText(
#                 vis_frame,
#                 f'ID Switches: {id_switches} | Ghost Tracks Filtered: {ghost_tracks_filtered}',
#                 (10, 90),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 255),
#                 2
#             )
            
#             # Write frame to video
#             writer.write(vis_frame)
            
#             # Save keyframes
#             if frame_count % 300 == 0:  # Save every 300 frames
#                 frame_path = os.path.join(output_dir, f'frame_{frame_count:06d}.jpg')
#                 cv2.imwrite(frame_path, vis_frame)
        
#         # Log data to CSV
#         with open(csv_path, 'a', newline='') as f:
#             writer_csv = csv.writer(f)
#             writer_csv.writerow([
#                 frame_count,
#                 len(raw_detections),
#                 len(tracks),
#                 new_left_count, 
#                 new_right_count,
#                 len(tracker.counted_left), 
#                 len(tracker.counted_right),
#                 len(current_left_ids),
#                 len(current_right_ids),
#                 id_switches,
#                 ghost_tracks_filtered
#             ])
        
#         # Print progress
#         if frame_count % 100 == 0:
#             elapsed_time = time.time() - start_time
#             fps_processing = frame_count / elapsed_time
#             print(f"Processed {frame_count}/{total_frames} frames. Processing FPS: {fps_processing:.1f}")
#             print(f"Raw detections: {len(raw_detections)}, Tracked: {len(tracks)}")
#             print(f"Unique people counted - Left: {len(tracker.counted_left)}, Right: {len(tracker.counted_right)}")
#             print(f"ID switches: {id_switches}, Ghost tracks filtered: {ghost_tracks_filtered}")
    
#     # Cleanup
#     cap.release()
#     if output_video:
#         writer.release()
    
#     # Final stats
#     elapsed_time = time.time() - start_time
#     print(f"Video processing complete in {elapsed_time:.2f} seconds")
#     print(f"Output saved to: {output_dir}")
#     print(f"Final counts - Left zone: {len(tracker.counted_left)}, Right zone: {len(tracker.counted_right)}")
#     print(f"Total ID switches: {id_switches}, Ghost tracks filtered: {ghost_tracks_filtered}")
    
#     return len(tracker.counted_left), len(tracker.counted_right)

# def main():
#     video_directory = "/media/chan/backup_SSD2/ASPED.c/Session_02152024/IntersectionB/Video/gopro04/"
#     output_base_dir = "merged_tracker_output/"
    
#     for root, _, files in os.walk(video_directory):
#         for file in files:
#             if file.endswith('GH010015.MP4'):
#                 video_path = os.path.join(root, file)
#                 video_output_dir = os.path.join(output_base_dir, os.path.splitext(file)[0])
                
#                 print(f'Processing {file}...')
#                 print(f'Output will be saved to: {video_output_dir}')
#                 start_time = time.time()
                
#                 left_count, right_count = process_video(video_path, video_output_dir)
                
#                 elapsed_time = time.time() - start_time
#                 print(f'Finished processing {file}')
#                 print(f'Unique pedestrians - Left zone: {left_count}, Right zone: {right_count}')
#                 print(f'Processing time: {elapsed_time:.2f} seconds')
#                 print('-' * 50)

# if __name__ == "__main__":
#     main()