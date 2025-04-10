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

   zoneA = np.array([[421, 155], [243, 263], [2652, 267], [2373, 124]], np.int32)						
   zoneB = np.array([[2672, 267], [2393, 124], [3263, 120], [3327,257]], np.int32)						
   zoneC = np.array([[250, 296],[250,1300], [3323, 1300], [3317, 283]], np.int32) 						
   return zoneA, zoneB, zoneC

def process_video(video, output_video=True):

    session="Session_12032024"
    output_dir = f'/home/schivilkar/dev/processed_video/{session}/IntersectionC/{video}'
    output_filename = os.path.join(output_dir, video+"_CROPPED.MP4")

    
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
    
    zoneA, zoneB, zoneC = setup_counting_zones(frame_width, frame_height)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if output_video:
            vis_frame = frame.copy()
            cv2.polylines(vis_frame, [zoneA], True, (255, 0, 0), 2) 
            cv2.polylines(vis_frame, [zoneB], True, (0, 0, 255), 2)
            cv2.polylines(vis_frame, [zoneC], True, (0, 255, 0), 2)

            frame_path = os.path.join(output_dir, 'draw_zones.jpg')
            cv2.imwrite(frame_path, vis_frame) 
        
    cap.release()
    
    
def main():
    video_list = ["GH010010"]
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
