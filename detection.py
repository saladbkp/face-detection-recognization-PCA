import cv2
import os
import json
import numpy as np
import argparse
from datetime import datetime

def get_next_face_id(output_dir, person_name):
    """
    Get the next available face ID by checking existing files
    
    Args:
        output_dir (str): Directory containing face images
        person_name (str): Name of the person
    
    Returns:
        int: Next available face ID
    """
    if not os.path.exists(output_dir):
        return 1
    
    max_id = 0
    pattern = f"{person_name}_face_"
    
    for filename in os.listdir(output_dir):
        if filename.startswith(pattern) and filename.endswith('.jpg'):
            try:
                # Extract the number from filename like "Joseph_Lai_face_0001.jpg"
                id_str = filename[len(pattern):filename.rfind('.jpg')]
                face_id = int(id_str)
                max_id = max(max_id, face_id)
            except ValueError:
                continue
    
    return max_id + 1

def detect_faces_in_video(video_path, output_dir, person_name):
    """
    Detect faces in video using OpenCV Haar Cascade and save extracted faces
    
    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save extracted faces
        person_name (str): Name of the person in the video
    
    Returns:
        dict: Metadata about detected faces
    """
    # Initialize face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the starting face ID to avoid conflicts
    starting_face_id = get_next_face_id(output_dir, person_name)
    
    # Initialize metadata
    metadata = {
        'video_name': os.path.basename(video_path),
        'person_name': person_name,
        'detection_timestamp': datetime.now().isoformat(),
        'faces': []
    }
    
    frame_count = 0
    face_count = 0
    
    print(f"Processing video: {video_path}")
    print(f"Saving faces to: {output_dir}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face_count += 1
            current_face_id = starting_face_id + face_count - 1
            
            # Extract face region with some padding
            padding = 20
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)
            
            face_img = frame[y_start:y_end, x_start:x_end]
            
            # Resize face to standard size (100x100)
            face_resized = cv2.resize(face_img, (100, 100))
            
            # Save face image with unique ID
            face_filename = f"{person_name}_face_{current_face_id:04d}.jpg"
            face_path = os.path.join(output_dir, face_filename)
            cv2.imwrite(face_path, face_resized)
            
            # Add face metadata
            face_metadata = {
                'face_id': current_face_id,
                'frame_number': frame_count,
                'filename': face_filename,
                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'face_size': {'width': 100, 'height': 100}
            }
            metadata['faces'].append(face_metadata)
        
        # Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames, found {face_count} faces")
    
    cap.release()
    
    print(f"\nDetection completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total faces detected: {face_count}")
    
    return metadata

def save_face_metadata(metadata, output_dir):
    """
    Save face detection metadata to JSON file
    
    Args:
        metadata (dict): Face detection metadata
        output_dir (str): Directory to save metadata file
    """
    video_name = metadata['video_name'].split('.')[0]
    metadata_filename = f"{video_name}_metadata.json"
    metadata_path = os.path.join(output_dir, metadata_filename)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Metadata saved to: {metadata_path}")

def main():
    """
    Main function to run face detection with command line argument support
    """
    parser = argparse.ArgumentParser(description='Face Detection System')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--person', type=str, default='Joseph_Lai', help='Name of the person in the video')
    parser.add_argument('--output', type=str, default='faces', help='Output directory for extracted faces')
    
    args = parser.parse_args()
    
    # Default configuration if no arguments provided
    if args.video is None:
        video_path = "videos/Joseph_Lai.mp4"
    else:
        video_path = args.video
    
    output_dir = args.output
    person_name = args.person
    
    try:
        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Processing video: {video_path}")
        print(f"Person name: {person_name}")
        print(f"Output directory: {output_dir}")
        
        # Detect faces and extract metadata
        metadata = detect_faces_in_video(video_path, output_dir, person_name)
        
        # Save metadata to JSON file
        save_face_metadata(metadata, output_dir)
        
        print("\n=== Face Detection Summary ===")
        print(f"Video: {metadata['video_name']}")
        print(f"Person: {metadata['person_name']}")
        print(f"Total faces detected: {len(metadata['faces'])}")
        print(f"Faces saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during face detection: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main()