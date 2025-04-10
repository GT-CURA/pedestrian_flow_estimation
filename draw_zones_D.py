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

def setup_counting_zones(width, height, zone_config=None):
    
    zoneA = np.array([[250, 247], [480, 193], [1118, 367], [602, 492]], dtype=np.int32)
    zoneC = np.array([[180, 596], [1210, 360], [2050, 593], [1500, 1200], [387, 1380]], dtype=np.int32)
    zoneD = np.array([[1772, 958], [2416, 209], [3025, 261], [3024, 967]], dtype=np.int32)
    zoneE = np.array([[1121, 330], [2038, 550], [2508, 32], [2066, 9]], dtype=np.int32)
    return zoneA, zoneD, zoneC, zoneE


def process_video(video, output_video=True):

    session="Session_12032024"
    output_dir = f'/home/schivilkar/dev/processed_video/{session}/IntersectionD/{video}'
    
    os.makedirs(output_dir, exist_ok=True)
    video_name = video+"_CROPPED.MP4"
    video_path = os.path.join(output_dir, video_name)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frame count:", total_frames)
    print(frame_width)
    print(frame_height)
    
    zoneA, zoneD, zoneC, zoneE = setup_counting_zones(frame_width, frame_height)
    
    # Initialize counters and tracking variables
    frame_count = 0
    start_time = time.time()
    
    # Main processing loop
    while True and frame_count < 400:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if output_video:
            vis_frame = frame.copy()
            # Draw zones
            cv2.polylines(vis_frame, [zoneA], True, (255, 0, 0), 2)  # Zone A in blue
            cv2.polylines(vis_frame, [zoneD], True, (0, 0, 255), 2)  # Zone D in red
            cv2.polylines(vis_frame, [zoneC], True, (0, 255, 0), 2)  # Zone C in green
            cv2.polylines(vis_frame, [zoneE], True, (255, 255, 0), 2)  # Zone E

            frame_path = os.path.join(output_dir, 'draw_zones.jpg')
            cv2.imwrite(frame_path, vis_frame) 
        
    cap.release()
    
    

def main():
    video_list = ["GH010019"]
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
