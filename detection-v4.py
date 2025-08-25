import cv2
import json
import os
import numpy as np
import argparse
from datetime import datetime

def detect_faces_and_save_data(video_path, output_face_dir, output_json_path):
    """
    Detect faces in video using Haarcascade and save face images and position data
    
    Args:
        video_path: Input video path
        output_face_dir: Directory to save face images
        output_json_path: Path to save JSON data
    """
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Create output directory
    if not os.path.exists(output_face_dir):
        os.makedirs(output_face_dir)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    frame_count = 0
    all_face_data = []
    face_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Save face image
            face_filename = f"face_{face_id:06d}_frame_{frame_count:06d}.jpg"
            face_path = os.path.join(output_face_dir, face_filename)
            cv2.imwrite(face_path, face_img)
            
            # Calculate timestamp
            timestamp = frame_count / fps if fps > 0 else 0
            
            # Save face data
            face_data = {
                "face_id": face_id,
                "frame_number": frame_count,
                "timestamp": timestamp,
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "center_x": int(x + w // 2),
                "center_y": int(y + h // 2),
                "area": int(w * h),
                "image_path": face_path,
                "image_filename": face_filename
            }
            all_face_data.append(face_data)
            face_id += 1
        
        frame_count += 1
        
        # Show progress
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    cap.release()
    
    # Save JSON data
    video_info = {
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "total_faces_detected": len(all_face_data),
        "processing_date": datetime.now().isoformat(),
        "faces": all_face_data
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(video_info, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetection completed!")
    print(f"Total faces detected: {len(all_face_data)}")
    print(f"Face images saved to: {output_face_dir}")
    print(f"JSON data saved to: {output_json_path}")
    
    return video_info

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect faces in video and save face images and data')
    parser.add_argument('--video', required=True, help='Input video file path')
    parser.add_argument('--person', required=True, help='Person name for organizing output directory')
    args = parser.parse_args()
    
    # Configure paths based on person name
    video_input = args.video
    output_face_directory = f"faces/lock_version/{args.person}"
    output_json_file = f"faces/lock_version/{args.person}/{args.person}_faces_detection.json"
    
    # Check if video file exists
    if not os.path.exists(video_input):
        print(f"Error: Video file {video_input} not found!")
        return
    
    # Execute face detection
    detect_faces_and_save_data(video_input, output_face_directory, output_json_file)

if __name__ == "__main__":
    main()