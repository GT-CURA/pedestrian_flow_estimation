import numpy as np
import cv2
import csv
import time # to track how long labeling took
import os

if __name__ == '__main__':

    start_time = time.time()
    output_video = True
    scaling_factor = 2
    
    # Define ROI (Region of Interest) coordinates 
    x_start = 500
    y_start = 200
    x_end = 1800
    y_end = 900

    crop_width = x_end - x_start
    crop_height = y_end - y_start

    
    filenames = ["GH010012"]
    video = "GH010012"
    session = "Session_02152024"  # Replace with actual session name
    output_dir = f'/home/chan/schivilkar/processed_video/{session}/IntersectionD/{video}'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, video+"_CROPPED7.MP4")
    
    # Loop through videos 
    for video in filenames:
        print(f"Start processing {video}...")
        video = "GH010012"
        session = "Session_02152024" 

        os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionD/Video/gopro03/{v}.MP4' -an -c:v copy -t 15 '{out_dir}/{v}_MUTED.MP4'".format(s = session, v=video, out_dir=output_dir))
    
        video_name = video +"_MUTED.MP4"
        video_path = os.path.join(output_dir, video_name)
        capture = cv2.VideoCapture(video_path) 


        # Get total number of frames in the video
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))    

        # Get the frame width and height
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Print the frame size
        print(f"Frame size: Width = {frame_width}, Height = {frame_height}")


        if output_video:
           
            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
            out = cv2.VideoWriter(output_filename, fourcc, 30.0, (crop_width*scaling_factor, crop_height*scaling_factor))  
  
        while True:
            # Read each frame
            _, im = capture.read()
            if im is None:
                break

            # Crop the frame to the defined ROI
            cropped_im = im[y_start:y_end, x_start:x_end]

            # Resize cropped image (upscaling)
            cropped_im = cv2.resize(cropped_im, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)

            if output_video:
                output_image_frame = cropped_im
                out.write(output_image_frame)
                

        capture.release()
        
        if output_video:
            out.release()
            cv2.destroyAllWindows()

        # Calculate and print total execution time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total time taken: {elapsed_time:.2f} seconds")