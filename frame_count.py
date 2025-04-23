import numpy as np
import cv2
import time 
import os

if __name__ == '__main__':

    
    filenames = ["GH090012_anglechanged"]
    
    # Loop through videos 
    for video in filenames:
        print(f"Start processing {video}...")
        session = "Session_10222024" 
        output_dir = f'/home/schivilkar/dev/processed_video/{session}/Path1/{video}'
        os.makedirs(output_dir, exist_ok=True)
        os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/Path1/Video/gopro06/{v}.MP4' -an -c:v copy '{out_dir}/{v}_MUTED.MP4'".format(s = session, v=video, out_dir=output_dir))
    
        video_name = video +"_MUTED.MP4"
        video_path = os.path.join(output_dir, video_name)
        capture = cv2.VideoCapture(video_path) 

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))    
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Frame size: Width = {frame_width}, Height = {frame_height}")
        print(f"frame count: {total_frames}")
        while True:
            ret, frame = capture.read()
            if not ret:
                break
                
            frame_path = os.path.join(output_dir, 'draw_zones.jpg')
            vis_frame = frame.copy()
            cv2.imwrite(frame_path, vis_frame) 
        
        capture.release()
        end_time = time.time()
        print(f"Processing completed!!")