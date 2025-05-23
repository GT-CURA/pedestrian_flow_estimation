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

    zoneA = np.array([[406, 336], [791, 318],[1068, 397],[1050, 496], [130,577]], np.int32)
    zoneC = np.array([[132,589], [1060, 499],[1900, 800],[1910, 1000],[21,1000]], np.int32)
    zoneD = np.array([[1093, 387],[1075, 485],[1450,626], [1840, 486], [1657,408]], np.int32)
    return zoneA, zoneD, zoneC 

def process_video(video, output_video=True):

    session="Session_12032024"
    output_dir = f'/home/schivilkar/dev/processed_video/{session}/IntersectionB/{video}'
    
    os.makedirs(output_dir, exist_ok=True)
    #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionB/Video/gopro04/{v}.MP4' -ss 15 -t 15 -an -c:v copy '{out_dir}/{v}_MUTED15s.MP4'".format(s=session, v=video, out_dir=output_dir))

    #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionB/Video/gopro03/{v}.MP4' -an -c:v copy -t 15 '{out_dir}/{v}_MUTED15s.MP4'".format(s=session, v=video, out_dir=output_dir)) 
    
    video_name = video+"_MUTED15s.MP4"
    video_path = os.path.join(output_dir, video_name)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frame count:", total_frames)
    print(frame_width)
    print(frame_height)
    
    zoneA, zoneD, zoneC = setup_counting_zones(frame_width, frame_height)
    
    # Initialize counters and tracking variables
    frame_count = 0
    
    # Main processing loop
    while True:
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

            frame_path = os.path.join(output_dir, 'frame.jpg')
            cv2.imwrite(frame_path, vis_frame) 
        
    cap.release()
    
def main():
    video_list = ["GH010015"]

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
