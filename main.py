import numpy as np
import tracker
from detector import Detector
import cv2
import csv
import time # to track how long labeling took
import os

if __name__ == '__main__':

    start_time = time.time()
    output_video = True # True if you want to see demo, save demo as video file

    # Define frame size and upscaling factor
    img_width = 1920 # 960
    img_height = 1080 # 540
    scaling_factor = 2
    
    # Define ROI (Region of Interest) coordinates 
    x_start = 1180
    y_start = 370
    x_end = 1845
    y_end = 945

    crop_width = x_end - x_start
    crop_height = y_end - y_start

    ##################### MASKING POLYGON #####################
    # Create a polygon for collision line detection based on video dimensions
    mask_image_temp = np.zeros((crop_height, crop_width), dtype=np.uint8)

    # Initialize two polygons for collision detection
    # list_pts_blue = [[956, 440], [946, 525], [1149, 525], [1131, 440]] # 524 up-down
    # list_pts_blue = [[959, 290], [949, 350], [1120, 350], [1120, 290]] # 601 up-down
    # list_pts_blue = [[990, 250], [985, 310], [1150, 310], [1150, 250]] # 607 up-down
    # list_pts_blue = [[985, 130], [970, 185], [1130, 185], [1120, 130]] # 621 up-down
    # list_pts_blue = [[970, 170], [960, 220], [1125, 220], [1115, 170]] # 628 up-down
    list_pts_blue = [[1440, 485], [1675, 875], [1770, 850], [1518, 485]]  # 628 left-right
    
    # Shift the polygon to match ROI
    list_pts_blue = [[x - x_start, y - y_start] for x, y in list_pts_blue] 

    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # Fill the second polygon
    mask_image_temp = np.zeros((crop_height, crop_width), dtype=np.uint8)
    # list_pts_yellow = [[935, 535], [910, 630], [1165, 630], [1150, 535]] # 524
    # list_pts_yellow = [[950, 370], [915, 425], [1145, 425], [1126, 369]] # 601
    # list_pts_yellow = [[985, 330], [960, 385], [1165, 385], [1165, 330]] # 607
    # list_pts_yellow = [[965, 210], [935, 255], [1150, 255], [1140, 210]] # 621
    # list_pts_yellow = [[960, 235], [935, 280], [1150, 280], [1145, 235]] # 628 up-down
    list_pts_yellow = [[1350, 485], [1560, 890], [1665, 875], [1430, 485]] # 628 left-right
   
    list_pts_yellow = [[x - x_start, y - y_start] for x, y in list_pts_yellow]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # Mask for collision detection, containing two polygons (values range 0, 1, 2)
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # Resize polygons to match the scaled-up dimensions of the cropped image
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, 
                                          (crop_width * scaling_factor, crop_height * scaling_factor), 
                                          interpolation=cv2.INTER_NEAREST)

    # List overlapping with polygon
    list_overlapping_blue_polygon = []
    list_overlapping_yellow_polygon = []
    ###############################################################

    # Initialize YOLOv7
    detector = Detector()

    filenames = ["6282023_video1", "6282023_video2","6282023_video3","6282023_video4","6282023_video5","6282023_video6","6282023_video7","6282023_video8",
                "6282023_video9","6282023_video10","6282023_video11"]
    session = "Session_6282023"

    # Initialize cumulative frame number, up_count, and down_count
    cumulative_frame_number = 0
    cumulative_up_count = 0
    cumulative_down_count = 0

    # Open CSV file and write the header
    output_file = "6282023_lr_flow.csv"
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame_Number', 'Incremental_LEFT', 'Incremental_RIGHT', 'Total_LEFT', 'Total_RIGHT'])  

    # Loop through videos 
    for video in filenames:
        print(f"Start processing {video}...")

        # Initialize incremental counts for UP and DOWN
        incremental_up = 0
        incremental_down = 0
        #/media/chan/backup_SSD2/ASPED.c/Session_02152024/IntersectionB/Video/gopro04/

        
        new_path = '/media/backup_SSD/ASPEDv1_Video/{s}/Cadell/Video/Camera1_GTgopro05/{v}.MP4'.format(s = session, v=video)
        print("*********************")
        print(new_path)
        new_path2 = '/home/chan/dev/video_labeling/video/{s}/{v}_MUTED.MP4'.format(s = session, v=video)
        print(new_path2)

        output_dir = f'/home/chan/dev/video_labeling/video/{session}/{video}'
        os.makedirs(output_dir, exist_ok=True)

        # open video
        os.system("ffmpeg -i '/media/backup_SSD/ASPEDv1_Video/{s}/Cadell/Video/Camera1_GTgopro05/{v}.MP4' -an -c:v copy '{out_dir}/{v}_MUTED.MP4'".format(s = session, v=video, out_dir=output_dir))
        capture = cv2.VideoCapture("/home/chan/dev/video_labeling/video/{s}/{v}_MUTED.MP4".format(s = session,v= video)) 
  
        # Get total number of frames in the video
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))    

        # Get the frame width and height
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Print the frame size
        print(f"Frame size: Width = {frame_width}, Height = {frame_height}")


        if output_video:
            # color plate b,g,r
            blue_color_plate = [255, 0, 0]
            blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

            yellow_color_plate = [0, 255, 255]
            yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

            # Color image (values range 0-255)
            color_polygons_image = blue_image + yellow_image

            # Resize polygons to match the scaled-up dimensions of the cropped image
            color_polygons_image = cv2.resize(color_polygons_image, (crop_width*scaling_factor, crop_height*scaling_factor), interpolation = cv2.INTER_NEAREST)

            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
            out = cv2.VideoWriter('output_lr_demo_upscaled.mp4', fourcc, 30.0, (crop_width*scaling_factor, crop_height*scaling_factor))  

            # Set the position for drawing text, relative to the cropped video.
            font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
            draw_text_postion = (int(crop_width * 0.01), int(crop_height * 0.1))

        while True:
            # Read each frame
            _, im = capture.read()
            if im is None:
                break

            cumulative_frame_number += 1
            
            # Reset incremental up and down for the new frame
            incremental_up = 0
            incremental_down = 0

            # Crop the frame to the defined ROI
            cropped_im = im[y_start:y_end, x_start:x_end]

            # Resize cropped image (upscaling)
            cropped_im = cv2.resize(cropped_im, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)

            list_bboxs = []
            bboxes = detector.detect(cropped_im)

            if len(bboxes) > 0:
                # Update tracker with detected bounding boxes
                list_bboxs = tracker.update(bboxes, cropped_im)

            if output_video:
                if len(bboxes) > 0:
                    # Draw bounding boxes
                    output_image_frame = tracker.draw_bboxes(cropped_im, list_bboxs, line_thickness=None)
                else:
                    output_image_frame = cropped_im

                # Combine with polygon mask
                output_image_frame = cv2.add(output_image_frame, color_polygons_image)

            if len(list_bboxs) > 0:
                for item_bbox in list_bboxs:
                    x1, y1, x2, y2, label, track_id = item_bbox

                    x = int((x1 + x2)/2)
                    y = y2

                    # y1_offset = int(y1 + ((y2 - y1) * 0.6))
                    # y, x = y1_offset, x1

                    if polygon_mask_blue_and_yellow[y, x] == 1:
                        if track_id not in list_overlapping_blue_polygon:
                            list_overlapping_blue_polygon.append(track_id)

                        if track_id in list_overlapping_yellow_polygon:
                            cumulative_up_count += 1
                            incremental_up += 1
                            # print(f'Category: {label} | id: {track_id} | Exit collision | Total exit collisions: {up_count} | Exit id list: {list_overlapping_yellow_polygon}')
                            list_overlapping_yellow_polygon.remove(track_id)

                    elif polygon_mask_blue_and_yellow[y, x] == 2:
                        if track_id not in list_overlapping_yellow_polygon:
                            list_overlapping_yellow_polygon.append(track_id)

                        if track_id in list_overlapping_blue_polygon:
                            cumulative_down_count += 1
                            incremental_down += 1
                            # print(f'Category: {label} | id: {track_id} | Entry collision | Total entry collisions: {down_count} | Entry id list: {list_overlapping_blue_polygon}')
                            list_overlapping_blue_polygon.remove(track_id)

                # Removing IDs that are no longer found in polygons
                list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
                for id1 in list_overlapping_all:
                    is_found = False
                    for _, _, _, _, _, bbox_id in list_bboxs:
                        if bbox_id == id1:
                            is_found = True
                            break

                    if not is_found:
                        if id1 in list_overlapping_yellow_polygon:
                            list_overlapping_yellow_polygon.remove(id1)
                        if id1 in list_overlapping_blue_polygon:
                            list_overlapping_blue_polygon.remove(id1)
                list_overlapping_all.clear()
                list_bboxs.clear()
            else:
                # Clear the lists for the next frame
                list_overlapping_blue_polygon.clear()
                list_overlapping_yellow_polygon.clear()

            # Append cumulative frame number, incremental UP/DOWN, and total counts to CSV
            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([cumulative_frame_number, incremental_down, incremental_up, cumulative_down_count, cumulative_up_count])
            output_dir2 = "output_frames_polygon/"
            os.makedirs(output_dir2, exist_ok=True) 

            if cumulative_frame_number % 10 == 0:
                print(f"****************frame: {cumulative_frame_number} out of {total_frames} processed")
                frame_filename = os.path.join(output_dir2, f"frame_{cumulative_frame_number}.jpg")
                cv2.imwrite(frame_filename, output_image_frame)


            if output_video:
                text_draw = f'LEFT: {cumulative_down_count} , RIGHT: {cumulative_up_count}'
                output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                                org=draw_text_postion,
                                                fontFace=font_draw_number,
                                                fontScale=1, color=(255, 255, 255), thickness=2)

                out.write(output_image_frame)

                cv2.imshow('demo', output_image_frame)
                cv2.waitKey(1)

        capture.release()
        
        if output_video:
            out.release()
            cv2.destroyAllWindows()

        # Calculate and print total execution time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total time taken: {elapsed_time:.2f} seconds")