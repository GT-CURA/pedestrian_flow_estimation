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
        
        self.tracker = DeepSort(
            max_age=50,
            n_init=2,
            max_iou_distance=0.9,
            max_cosine_distance=0.4,
            nn_budget=100
        )
        
        self.track_history = defaultdict(list)
        self.max_trail_length = 30
        
        self.tracks_in_left_zone = set()
        self.tracks_in_right_zone = set()
        self.counted_left = set()
        self.counted_right = set()

    def detect_and_track(self, frame):
        results = self.model(frame, conf=0.4)[0]
        
        # Get raw detections for visualization
        raw_detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) == 0:  # Class 0 is person
                raw_detections.append((float(x1), float(y1), float(x2), float(y2), float(score)))
        
        # # Format detections for DeepSORT
        deepsort_detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) == 0:  # Class 0 is person
                deepsort_detections.append(([x1, y1, x2-x1, y2-y1], score, 'person'))
        
        
        # # Update tracks
        tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        
        # # Process tracks - get positions, update trails
        track_data = []
        for track in tracks:
            #print(vars(track))
        
            if not track.is_confirmed():
                continue
                
            track_id = int(track.track_id)
            ltrb = track.to_ltrb()
            
        #   # Get bounding box coordinates
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
        #     # Update trail for visualization
            self.track_history[track_id].append((center_x, center_y))
            if len(self.track_history[track_id]) > self.max_trail_length:
                self.track_history[track_id].pop(0)
            
        #     # Add to output data
            track_data.append({
                'id': track_id,
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'confidence': track.det_conf
            })
            
        return track_data, raw_detections
        #return raw_detections

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
    os.makedirs(output_dir, exist_ok=True)
    # session="Session_02152024"
    # video = "GH010015"
    # output_dir = f'/home/schivilkar/dev/video_labeling/video/{session}/{video}'
    # os.makedirs(output_dir, exist_ok=True)
    # #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionB/Video/gopro04/{v}.MP4' -an -c:v copy '{out_dir}/{v}_MUTED.MP4'".format(s = session, v=video, out_dir=output_dir))
    # #print("Video Path:**********", video_path)
   # os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionB/Video/gopro04/{v}.MP4' -an -c:v copy -t 30 '{out_dir}/{v}_MUTED_30s.MP4'".format(s=session, v=video, out_dir=output_dir))

    # # Setup video capture
    video_directory = "/home/schivilkar/dev/video_labeling/video/Session_02152024/GH010015"
    video_path1 = os.path.join(video_directory, "GH010015_MUTED_30s.MP4")
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
    
    # # Set up CSV for data logging
    csv_path = os.path.join(output_dir, 'pedestrian_counts.csv')
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
        
        # Visualization - if requested
        if output_video:
            # Create a copy for visualization
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
            
            # # SECOND: Draw tracks and trails
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
                
            #     # Draw trail
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
                len(current_right_ids)
            ])
        
        #Print progress
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            fps_processing = frame_count / elapsed_time
            print(f"Processed {frame_count}/{total_frames} frames. Processing FPS: {fps_processing:.1f}")
            print(f"Raw detections: {len(raw_detections)}, Tracked: {len(tracks)}")
            print(f"Unique people counted - Left: {len(tracker.counted_left)}, Right: {len(tracker.counted_right)}")
    
    # Cleanup
    cap.release()
    if output_video:
        writer.release()
    
    # Final stats
    elapsed_time = time.time() - start_time
    print(f"Video processing complete in {elapsed_time:.2f} seconds")
    print(f"Output saved to: {output_dir}")
    print(f"Final counts - Left zone: {len(tracker.counted_left)}, Right zone: {len(tracker.counted_right)}")
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
        
#         self.tracker = DeepSort(
#             max_age=50,
#             n_init=2,
#             max_iou_distance=0.9,
#             max_cosine_distance=0.4,
#             nn_budget=100
#         )
        
#         self.track_history = defaultdict(list)
#         self.max_trail_length = 30
        
#         self.tracks_in_left_zone = set()
#         self.tracks_in_right_zone = set()
#         self.counted_left = set()
#         self.counted_right = set()

#     def detect_and_track(self, frame):
#         results = self.model(frame, conf=0.5)[0]
        
#         # Get raw detections for visualization
#         raw_detections = []
#         for r in results.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = r
#             if int(class_id) == 0:  # Class 0 is person
#                 raw_detections.append((float(x1), float(y1), float(x2), float(y2), float(score)))
        
#         # Format detections for DeepSORT
#         deepsort_detections = []
#         for r in results.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = r
#             if int(class_id) == 0:  # Class 0 is person
#                 deepsort_detections.append(([x1, y1, x2-x1, y2-y1], score, 'person'))
        
#         # Update tracks
#         tracks = self.tracker.update_tracks(deepsort_detections, frame=frame)
        
#         # Process tracks - get positions, update trails
#         track_data = []
#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
                
#             track_id = int(track.track_id)
#             ltrb = track.to_ltrb()
            
#             # Get bounding box coordinates
#             x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)
            
#             # Update trail for visualization
#             self.track_history[track_id].append((center_x, center_y))
#             if len(self.track_history[track_id]) > self.max_trail_length:
#                 self.track_history[track_id].pop(0)
            
#             # Add to output data
#             track_data.append({
#                 'id': track_id,
#                 'bbox': (x1, y1, x2, y2),
#                 'center': (center_x, center_y),
#                 'confidence': track.det_conf
#             })
            
#         return track_data, raw_detections

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
#     # session="Session_02152024"
#     # video = "GH010015"
#     # output_dir = f'/home/schivilkar/dev/video_labeling/video/{session}/{video}'
#     # os.makedirs(output_dir, exist_ok=True)
#     # #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionB/Video/gopro04/{v}.MP4' -an -c:v copy '{out_dir}/{v}_MUTED.MP4'".format(s = session, v=video, out_dir=output_dir))
#     # #print("Video Path:**********", video_path)
#     # #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionB/Video/gopro04/{v}.MP4' -an -c:v copy -t 30 '{out_dir}/{v}_MUTED_30s.MP4'".format(s=session, v=video, out_dir=output_dir))

#     # # Setup video capture
#     video_directory = "/home/schivilkar/dev/video_labeling/video/Session_02152024/GH010015"
#     video_path1 = os.path.join(video_directory, "GH010015_MUTED_30s.MP4")
#     print(output_dir)
#     print(video_path)
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
#             'Current_Left_Zone', 'Current_Right_Zone'
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
    
#     # Main processing loop
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         frame_count += 1
        
#         # Get tracks with IDs and raw detections
#         tracks, raw_detections = tracker.detect_and_track(frame)
        
#         # Track zone occupancy for this frame
#         current_left_ids = set()
#         current_right_ids = set()
#         new_left_count = 0
#         new_right_count = 0
        
#         # Process each tracked pedestrian
#         for track_data in tracks:
#             track_id = track_data['id']
#             center_x, center_y = track_data['center']
            
#             # Check zone containment
#             in_left = cv2.pointPolygonTest(left_zone, (float(center_x), float(center_y)), False) >= 0
#             in_right = cv2.pointPolygonTest(right_zone, (float(center_x), float(center_y)), False) >= 0
            
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
            
#             # # FIRST: Draw ALL raw detections
#             for det in raw_detections:
#                 x1, y1, x2, y2, conf = det
#                 # Draw all raw detections in yellow
#                 cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
#                 cv2.putText(
#                     vis_frame,
#                     f"{conf:.2f}",
#                     (int(x1), int(y1) - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (0, 255, 255),
#                     1
#                 )
            
#             # SECOND: Draw tracks and trails
#             for track_data in tracks:
#                 track_id = track_data['id']
#                 x1, y1, x2, y2 = track_data['bbox']
                
#                 # Define color based on which zone the person is in
#                 if track_id in current_left_ids and track_id in current_right_ids:
#                     color = (255, 0, 255)  # Purple for overlap
#                 elif track_id in current_left_ids:
#                     color = (255, 0, 0)    # Blue for left zone
#                 elif track_id in current_right_ids:
#                     color = (0, 0, 255)    # Red for right zone
#                 else:
#                     color = (0, 255, 0)    # Green for no zone
                
#                 # Draw bounding box and ID
#                 cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(
#                     vis_frame, 
#                     f"ID: {track_id}", 
#                     (x1, y1-10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, 
#                     color, 
#                     2
#                 )
                
#                 # Draw trail
#                 points = np.array(tracker.track_history[track_id], np.int32)
#                 if len(points) > 1:
#                     points = points.reshape((-1, 1, 2))
#                     cv2.polylines(vis_frame, [points], False, color, 2)
            
#             # Add counters and frame number
#             cv2.putText(
#                 vis_frame,
#                 f'Frame: {frame_count} | Raw Detections: {len(raw_detections)}',
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
#                 len(current_right_ids)
#             ])
        
#         # Print progress
#         if frame_count % 100 == 0:
#             elapsed_time = time.time() - start_time
#             fps_processing = frame_count / elapsed_time
#             print(f"Processed {frame_count}/{total_frames} frames. Processing FPS: {fps_processing:.1f}")
#             print(f"Raw detections: {len(raw_detections)}, Tracked: {len(tracks)}")
#             print(f"Unique people counted - Left: {len(tracker.counted_left)}, Right: {len(tracker.counted_right)}")
    
#     # Cleanup
#     cap.release()
#     if output_video:
#         writer.release()
    
#     # Final stats
#     elapsed_time = time.time() - start_time
#     print(f"Video processing complete in {elapsed_time:.2f} seconds")
#     print(f"Output saved to: {output_dir}")
#     print(f"Final counts - Left zone: {len(tracker.counted_left)}, Right zone: {len(tracker.counted_right)}")
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