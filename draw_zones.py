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

    #Final zones 

    #For Intersection C
    # zone_A = np.array([[344, 537], [309, 649],[1242, 600],[1137, 517]], np.int32)
    # zoneD = np.array([[1159, 513], [1251, 603],[1824, 533],[1687,482]], np.int32)
    # zoneC = np.array([[297,685], [956, 642],[1584, 579],[1822,607], [1821, 946], [167,956], [208,743]], np.int32) 

    #For PATH1 and Path2
    #zone_A = np.array([[1100, 564], [628, 656],[581, 443],[826, 424]], np.int32)
    #zoneD = np.array([[628, 670], [1100, 580],[1716, 753],[734, 980]], np.int32)

    #For IntersectionD
    # zone_A = np.array([[301, 469], [457, 381],[749, 412],[480, 539]], np.int32)
    # zoneB = np.array([[500, 539], [1032, 688],[1346, 333],[1004, 294]], np.int32)
    # zoneC = np.array([[310,639], [470, 545],[984, 684],[681,981]], np.int32)
    # zoneD = np.array([[1072,687], [1365, 315],[1829, 341],[1805,785]], np.int32)   
    # zoneE = np.array([[1000,285], [1212, 310],[1444, 172],[1249, 164]], np.int32)  

def setup_counting_zones(width, height, zone_config=None):
    # zone_A = np.array([[301, 469], [457, 381],[749, 412],[480, 539]], np.int32)
    # zoneB = np.array([[500, 539], [1032, 688],[1346, 333],[1004, 294]], np.int32)
    # # zoneC = np.array([[313,627], [651, 465],[950, 505],[1066,658], [703,957]], np.int32)
    # # zoneD = np.array([[965, 508],[1072,653], [1775, 748],[1764, 339],[1237,321]], np.int32)   
    # zoneE = np.array([[571,389], [1488, 460],[1595, 116],[1175, 85], [583,374]], np.int32)   

    # # zone_A = np.array([[511, 371], [788, 353],[1104, 402],[1098, 491], [286,560]], np.int32)
    # # zoneC = np.array([[278,578], [1086, 486],[1405, 611],[1837,646], [1828, 949], [53,945]], np.int32)
    # # zoneD = np.array([[1124, 401],[1119, 487],[1474,603],[1839,640], [1839,431], [1687, 388], [1116,400]], np.int32)  

    # zone_A = np.array([[7, 459], [82, 402],[458, 424],[42, 653]], np.int32)
    # #zoneD = np.array([[1251,615], [1465, 236],[1828, 238],[1829,869], [1242, 960], [1213,516]], np.int32)
    # zoneC = np.array([[19,704], [200, 1300],[1100, 1050],[1384,488], [548, 414]], np.int32)  
    # zoneD = np.array([[1125, 1000], [1601, 241],[2000, 100],[1900,1000]], np.int32)


    zone_A = np.array([[1615, 193], [1615, 327],[21, 382],[148, 245]], np.int32)
    zoneB = np.array([[1620, 192], [1623, 317],[3032, 175],[2755, 112]], np.int32)
    zoneC = np.array([[500, 880],[2965,800], [2962, 230], [550, 527]], np.int32) 
    return zone_A, zoneB, zoneC #, zoneD, zoneE

def process_video(video, output_video=True):

    session="Session_02152024"
    #output_dir = f'/home/schivilkar/dev/processed_video/{session}/IntersectionD/{video}'
    output_dir = f'/home/schivilkar/dev/processed_video/{session}/IntersectionC/{video}'
    os.makedirs(output_dir, exist_ok=True)
    #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionC/Video/gopro08/{v}.MP4' -an -c:v copy -t 15 '{out_dir}/{v}_MUTED15s.MP4'".format(s = session, v=video, out_dir=output_dir))
    #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionB/Video/gopro04/{v}.MP4' -an -c:v copy -t 15 '{out_dir}/{v}_MUTED15s.MP4'".format(s=session, v=video, out_dir=output_dir)) 
    #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/Path1/Video/gopro06/{v}.MP4' -an -c:v copy -t 15 '{out_dir}/{v}_MUTED15s.MP4'".format(s = session, v=video, out_dir=output_dir))
    #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/Path2/Video/gopro07/{v}.MP4' -an -c:v copy -t 15 '{out_dir}/{v}_MUTED15s.MP4'".format(s = session, v=video, out_dir=output_dir))
    #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionD/Video/gopro03/{v}.MP4' -an -c:v copy -t 15 '{out_dir}/{v}_MUTED15s.MP4'".format(s = session, v=video, out_dir=output_dir))
    video_name = video+"_CROPPED.MP4"
    #csv_name = video+"full_pedestrian_flow.csv"
    video_path = os.path.join(output_dir, video_name)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frame count:", total_frames)
    
    #zoneA, zoneB, zoneC, zoneD, zoneE = setup_counting_zones(frame_width, frame_height)
    zoneA, zoneB, zoneC = setup_counting_zones(frame_width, frame_height)
    
    # Initialize counters and tracking variables
    frame_count = 0
    start_time = time.time()
    
    # Main processing loop
    while True and frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if output_video:
            vis_frame = frame.copy()
            # Draw zones
            cv2.polylines(vis_frame, [zoneA], True, (255, 0, 0), 2)  # Zone A in blue
            cv2.polylines(vis_frame, [zoneB], True, (0, 0, 255), 2)  # Zone D in red
            cv2.polylines(vis_frame, [zoneC], True, (0, 255, 0), 2)  # Zone C in green
            # cv2.polylines(vis_frame, [zoneD], True, (0, 255, 255), 2)  # Zone D in red
            # cv2.polylines(vis_frame, [zoneE], True, (255, 255, 0), 2)  # Zone C in green

            frame_path = os.path.join(output_dir, 'frame.jpg')
            cv2.imwrite(frame_path, vis_frame) 
        
    cap.release()
    
    

def main():
    #video_list = ["GH010015"]
    #video_list = ["GH010006"]
    #video_list = ["GH010011"]
    #video_list = ["GH030006"]
    video_list = ["GH010006"]

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
