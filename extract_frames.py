import os
import cv2

def extract_first_frame(sessions, output_root):
    """
    Extract first frame from videos from the nested directory structure
    Args:
    sessions (list): List of Session Ids
    output_root (str): Root directory to save extracted frames
    """
    os.makedirs(output_root, exist_ok=True)
    
    # Process each session directory
    for session in sessions:
        # Validate session path
        base_path = "/media/chan/backup_SSD2/ASPED.c/"
        session_path = os.path.join(base_path, session)

        if not os.path.isdir(session_path):
            print(f"Warning: {session_path} is not a valid directory. Skipping.")
            continue
        
        session_name = os.path.basename(session_path)
        session_output_path = os.path.join(output_root, session_name)
        os.makedirs(session_output_path, exist_ok=True)
        
        # Iterate through intersection/path directories
        for intersection_dir in os.listdir(session_path):
            intersection_path = os.path.join(session_path, intersection_dir)
            
            # Check if it's a directory
            if not os.path.isdir(intersection_path):
                continue
            
            # Look for Video directory
            video_dir_path = os.path.join(intersection_path, 'Video')
            if not os.path.isdir(video_dir_path):
                continue
            
            # Find gopro directories
            for gopro_dir in os.listdir(video_dir_path):
                # Check if directory starts with 'gopro'
                if not gopro_dir.lower().startswith('gopro'):
                    continue
                
                gopro_full_path = os.path.join(video_dir_path, gopro_dir)
                
                # Find first .MP4 video
                video_found = False
                for video_file in os.listdir(gopro_full_path):
                    if video_file.upper().endswith('.MP4'):
                        full_video_path = os.path.join(gopro_full_path, video_file)
                        
                        try:
                            # Open the video
                            cap = cv2.VideoCapture(full_video_path)
                            
                            # Read the first frame
                            ret, frame = cap.read()
                            
                            if ret:
                                # Create output filename
                                output_filename = os.path.join(session_output_path, 
                                                              f"{intersection_dir}_{gopro_dir}_first_frame.jpg")
                                
                                cv2.imwrite(output_filename, frame)
                                print(f"Extracted first frame: {output_filename}")
                                
                                video_found = True
                            
                            # Release the video capture object
                            cap.release()
                            
                            # Break after first .MP4 video
                            if video_found:
                                break
                        
                        except Exception as e:
                            print(f"Error processing {full_video_path}: {e}")
                
                if video_found:
                    break

def main():
    sessions = ["Session_02152024", "Session_02292024", "Session_10222024", "Session_10292024", "Session_12032024"]
    # Output root directory
    output_root = "/home/schivilkar/dev/frame_extraction/"
    
    # Extract first frames
    extract_first_frame(sessions, output_root)

if __name__ == "__main__":
    main()