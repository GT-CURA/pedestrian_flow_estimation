import numpy as np
import cv2
import time 
import os

if __name__ == '__main__':

    start_time = time.time()
    output_video = True
    scaling_factor = 2
    
    x_start = 450
    y_start = 225
    x_end = 1750
    y_end = 850


    crop_width = x_end - x_start
    crop_height = y_end - y_start

    #"GH010013","GH020013", "GH030013","GH040013",
                #  "GH050013","GH060013",
                #  "GH070013","GH080013",
                #  "GH090013","GH100013",
                #  "GH110013", ,"GH140013", "GH150013""GH110013","GH130013"
                # ,"GH020016", "GH030016","GH040016",
                #  "GH050016","GH060016",
                #  "GH070016","GH080016",
                #  "GH090016"

    filenames = ["GH010016"]   
    
    # Loop through videos 
    for video in filenames:
        print(f"Start processing {video}...")
        session = "Session_02292024" 
        output_dir = f'/home/schivilkar/dev/processed_video/{session}/IntersectionD/{video}'
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, video+"_CROPPED.MP4")

        os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionD1/Video/gopro04/{v}.MP4' -an -c:v copy '{out_dir}/{v}_MUTED.MP4'".format(s = session, v=video, out_dir=output_dir))
    
        video_name = video +"_MUTED.MP4"
        video_path = os.path.join(output_dir, video_name)
        capture = cv2.VideoCapture(video_path) 

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))    
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Frame size: Width = {frame_width}, Height = {frame_height}")


        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
            out = cv2.VideoWriter(output_filename, fourcc, 30.0, (crop_width*scaling_factor, crop_height*scaling_factor))  
  
        count = 1
        while True:
            _, im = capture.read()
            if im is None:
                break
            
            count = count + 1
            # Crop the frame to the defined ROI
            cropped_im = im[y_start:y_end, x_start:x_end]

            cropped_im = cv2.resize(cropped_im, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)

            if output_video:
                output_image_frame = cropped_im
                out.write(output_image_frame)
                frame_path = os.path.join(output_dir, 'draw_frame.jpg')
                cv2.imwrite(frame_path, output_image_frame) 
                

        capture.release()
        
        if output_video:
            out.release()
            cv2.destroyAllWindows()
        
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"Removed intermediate file: {video_path}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total time taken: {elapsed_time:.2f} seconds")