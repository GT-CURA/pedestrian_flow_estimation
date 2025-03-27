import os
import cv2
import numpy as np

def mute_video(input_video, output_video):
    """ Mute the given video using ffmpeg. """
    print("entered mute function")
    print(input_video)
    print(output_video)
    os.system(f"ffmpeg -i '{input_video}' -an -c:v copy '{output_video}'")

    #os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionD/Video/gopro03/{v}.MP4' -an -c:v copy '{out_dir}/{v}_MUTED.MP4'".format(s = session, v=video, out_dir=output_dir))


def crop_video(input_video, output_video, x, y, w, h):
    """ Crop the video based on the given ROI coordinates and save it. """
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop frame
        roi = frame[y:y+h, x:x+w]
        
        # Write cropped frame
        out.write(roi)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Cropped video saved at: {output_video}")

def main():
    session = "Session_02152024"  # Replace with actual session name
    video = "GH010012"      # Replace with actual video name
    output_dir = f'/home/schivilkar/dev/processed_video/{session}/IntersectionD/{video}'
    os.makedirs(output_dir, exist_ok=True)
    os.system("ffmpeg -i '/media/chan/backup_SSD2/ASPED.c/{s}/IntersectionD/Video/gopro03/{v}.MP4' -an -c:v copy -t 15 '{out_dir}/{v}_MUTED.MP4'".format(s = session, v=video, out_dir=output_dir))
    
    #Define ROI coordinates (modify as needed)
    
    
    # Define the destination points (rectangular ROI after transformation)

    # Compute the perspective transform matrix

    #x, y, w, h = 100, 100, 500, 400
    print("Helllllooo")

    video_name = video +"_MUTED.MP4"
    video_path = os.path.join(output_dir, video_name)
    cap = cv2.VideoCapture(video_path)    
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    output_filename = os.path.join(output_dir, video+"_CROPPED4.MP4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    img_width = 1920 # 960
    img_height = 1080 # 540

    
    writer = cv2.VideoWriter(
        output_filename,
        fourcc,
        fps,
        (frame_width, frame_height)
    )

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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop frame
        cropped_im = frame[y_start:y_end, x_start:x_end]

            # Resize cropped image (upscaling)
        cropped_im = cv2.resize(cropped_im, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)

        
        # Write cropped frame
        writer.write(cropped_im)

    # Release resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
