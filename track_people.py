import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime
from collections import defaultdict
import time

def main():
    model = YOLO('yolov8n.pt')
    
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.3,
        nn_budget=100
    )
    
    input_source = "/media/chan/backup_SSD2/ASPED.c/Session_02152024/IntersectionB/Video/gopro04/GH010015.MP4"
    cap = cv2.VideoCapture(input_source)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'newtrack_{timestamp}.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_filename,
        fourcc,
        input_fps,
        (frame_width, frame_height)
    )
    
    track_history = defaultdict(lambda: [])
    max_trail_length = 30
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Detect objects using YOLO
        results = model(frame)[0]
        
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if int(class_id) == 0 and score > 0.2:
                detections.append(([x1, y1, x2-x1, y2-y1], score, 'person'))
        
        # Update tracks
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # Draw tracks and trails
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = int(track.track_id)
            ltrb = track.to_ltrb()
            
            # Get center point
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Update trail
            track_history[track_id].append((center_x, center_y))
            if len(track_history[track_id]) > max_trail_length:
                track_history[track_id].pop(0)
            
            # Draw trail
            points = np.array(track_history[track_id], np.int32)
            points = points.reshape((-1, 1, 2))
            if len(points) > 1:
                # Generate color using integer track_id
                color = (
                    # int((track_id * 123) % 255),
                    # int((track_id * 50) % 255),
                    # int((track_id * 200) % 255)
                    0, 255, 0
                )
                
                # Draw lines with proper thickness
                for i in range(len(points) - 1):
                    thickness = 10
                    try:
                        pt1 = tuple(map(int, points[i][0]))
                        pt2 = tuple(map(int, points[i+1][0]))
                        cv2.line(frame, pt1, pt2, color, thickness)
                    except Exception as e:
                        print(f"Error drawing line: {e}")
                        continue
            
            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, 
                f"ID: {track_id}", 
                (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 255, 0), 
                2
            )
        
        # Add frame counter
        cv2.putText(
            frame,
            f'Frame: {frame_count}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Write frame
        out.write(frame)
        
        # Print progress
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"Processed {frame_count} frames. FPS: {fps:.1f}")
    
    # Cleanup
    cap.release()
    out.release()
    print(f"Video processing complete. Output saved as: {output_filename}")

if __name__ == "__main__":
    main()

