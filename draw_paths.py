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

    zoneA = np.array([[155, 618], [1200, 376],[300, 159],[1, 215]], np.int32)
    zoneB = np.array([[155, 635], [1200, 400],[2950, 700],[3000, 1450],[450, 1450]], np.int32)
    return zoneA, zoneB

def process_video(video, output_video=True):

    session="Session_10222024"
    output_dir = f'/home/schivilkar/dev/processed_video/{session}/Path2/{video}'
    os.makedirs(output_dir, exist_ok=True)
    # #output_dir = f'/home/schivilkar/dev/processed_video/{session}/Path1/{video}'
    os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/Path2/Video/gopro07/{v}.MP4' -an -c:v copy '{out_dir}/{v}_MUTED.MP4'".format(s=session, v=video, out_dir=output_dir))

    video_name = video+"_MUTED.MP4"

    video_path = os.path.join(output_dir, video_name)
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frame count:", total_frames)
    print(frame_width)
    print(frame_height)
    
    # zoneA, zoneB = setup_counting_zones(frame_width, frame_height)
    
    # # Initialize counters and tracking variables
    # frame_count = 0
    # start_time = time.time()
    
    # # Main processing loop
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
            
    #     frame_count += 1
        
    #     if output_video:
    #         vis_frame = frame.copy()
    #         # Draw zones
    #         cv2.polylines(vis_frame, [zoneA], True, (255, 0, 0), 2)  # Zone A in blue
    #         cv2.polylines(vis_frame, [zoneB], True, (0, 0, 255), 2)  # Zone D in red
    #         frame_path = os.path.join(output_dir, 'draw_zones.jpg')
    #         cv2.imwrite(frame_path, vis_frame) 
        
    # cap.release()
    
    

def main():
    
    video_list = ["GH100009_anglechanged"]
    
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
