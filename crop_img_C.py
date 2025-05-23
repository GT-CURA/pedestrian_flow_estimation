import numpy as np
import cv2
import time 
import os

if __name__ == '__main__':

    start_time = time.time()
    output_video = True
    scaling_factor = 2
    
    # Define ROI (Region of Interest) coordinates 

    x_start = 200
    y_start = 400
    x_end = 1800
    y_end = 900

    crop_width = x_end - x_start
    crop_height = y_end - y_start

    
    #filenames = ["GH010006", "GH020006","GH030006","GH040006","GH050006","GH060006","GH070006","GH080006","GH090006"]
    #"GH010008",
    filenames = ["GH010061",
                 "GH020061",
                 "GH030061",
                 "GH040061",
                 "GH050061",
                 "GH060061",
                 "GH070061",
                 "GH080061",
                 "GH090061",
                 "GH100061"]
    # filenames = ["GH070009",
    #              "GH080009",
    #              "GH090009",
    #              "GH100009"]
    # Loop through videos 
    for video in filenames:
        print(f"Start processing {video}...")
        session = "Session_02292024" 
        output_dir = f'/home/schivilkar/dev/processed_video/{session}/IntersectionC/{video}'
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, video+"_CROPPED.MP4")
        input_dir = f'/home/schivilkar/dev/final_video_processing/{session}/IntersectionC/{video}'

        #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionC/Video/gopro08/{v}.MP4' -an -c:v copy '{out_dir}/{v}_MUTED.MP4'".format(s = session, v=video, out_dir=output_dir))
    
        video_name = video +"_MUTED.MP4"
        video_path = os.path.join(input_dir, video_name)
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
            count = count + 1
            if im is None:
                break

            # Crop the frame to the defined ROI
            cropped_im = im[y_start:y_end, x_start:x_end]

            cropped_im = cv2.resize(cropped_im, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
            if output_video:
                vis_frame = im.copy()
                output_image_frame = cropped_im
                out.write(output_image_frame)
                frame_path = os.path.join(output_dir, 'draw_crop.jpg')
                cv2.imwrite(frame_path, output_image_frame) 
                
        capture.release()
        
        if output_video:
            out.release()
            cv2.destroyAllWindows()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total time taken: {elapsed_time:.2f} seconds")