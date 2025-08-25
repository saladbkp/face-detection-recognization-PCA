#!/usr/bin/env python3
"""
Face Recognition Pipeline Script

This script automates the complete face recognition pipeline:
1. Detection: Extract faces from video using detection-v2.py
2. Training: Train PCA model using train-v2.py
3. Recognition: Perform face recognition using scan-template-v2.py

Usage:
    python run_pipeline.py --video videos/Joseph_Lai.mp4 --person Joseph_Lai
    python run_pipeline.py --live --person Joseph_Lai
"""

import argparse
import subprocess
import sys
import os
import cv2
import time
from pathlib import Path

def run_command(command, description):
    """
    Execute a command and handle errors
    
    Args:
        command: List of command arguments
        description: Description of the command for logging
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(command)}")
    print()
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {command[0]}")
        print("Please ensure Python is installed and accessible.")
        return False

def check_file_exists(file_path, description):
    """
    Check if a file exists
    
    Args:
        file_path: Path to the file
        description: Description of the file
    
    Returns:
        bool: True if file exists, False otherwise
    """
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} not found: {file_path}")
        return False

def record_video_from_camera(output_path, duration=10):
    """
    Record video from camera for specified duration
    
    Args:
        output_path: Path to save the recorded video
        duration: Recording duration in seconds (default: 10)
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"📹 Starting camera recording for {duration} seconds...")
    print("📷 Please look at the camera!")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return False
    
    # Get camera properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    start_time = time.time()
    frame_count = 0
    
    print(f"🔴 Recording started... ({duration}s countdown)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to read frame from camera")
            break
        
        # Calculate remaining time
        elapsed_time = time.time() - start_time
        remaining_time = duration - elapsed_time
        
        if remaining_time <= 0:
            break
        
        # Add countdown text to frame
        countdown_text = f"Recording: {remaining_time:.1f}s"
        cv2.putText(frame, countdown_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Display frame (optional)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Recording completed! Saved {frame_count} frames to {output_path}")
    return True

def create_output_directory(person_name):
    """
    Create output directory for the person
    
    Args:
        person_name: Name of the person
    
    Returns:
        str: Path to the created directory
    """
    output_dir = f"faces/lock_version/{person_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory created/verified: {output_dir}")
    return output_dir

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Complete face recognition pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python run_pipeline.py --video videos/Joseph_Lai.mp4 --person Joseph_Lai
    python run_pipeline.py --live --person Joseph_Lai

This will:
1. Extract faces from the video and save to faces/lock_version/{person}/
2. Train a PCA model using the extracted faces
3. Perform face recognition on the video using template matching

Live mode:
- Records 10 seconds from camera
- Processes the recorded video for detection and training
- Uses live camera for real-time recognition (press 'q' to exit)
        """
    )
    
    # Create mutually exclusive group for video input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', help='Input video file path')
    input_group.add_argument('--live', action='store_true', help='Use live camera input')
    
    parser.add_argument('--person', required=True, help='Person name for organizing output')
    args = parser.parse_args()
    
    person_name = args.person
    is_live_mode = args.live
    
    if is_live_mode:
        video_path = f"temp_recording_{person_name}.mp4"
        print("🎥 Live camera mode activated!")
    else:
        video_path = args.video
    
    print("🚀 Face Recognition Pipeline Starting...")
    if is_live_mode:
        print(f"📹 Mode: Live Camera Recording")
    else:
        print(f"📹 Video: {video_path}")
    print(f"👤 Person: {person_name}")
    
    # Handle video input based on mode
    if is_live_mode:
        # Record video from camera
        if not record_video_from_camera(video_path, duration=10):
            print("\n❌ Pipeline aborted: Camera recording failed")
            sys.exit(1)
    else:
        # Check if input video exists
        if not check_file_exists(video_path, "Input video"):
            print("\n❌ Pipeline aborted: Input video not found")
            sys.exit(1)
    
    # Check if required scripts exist
    scripts = [
        ("detection-v2.py", "Face detection script"),
        ("train-v2.py", "Training script"),
        ("scan-template-v2.py", "Recognition script")
    ]
    
    for script, description in scripts:
        if not check_file_exists(script, description):
            print(f"\n❌ Pipeline aborted: {description} not found")
            sys.exit(1)
    
    # Create output directory
    output_dir = create_output_directory(person_name)
    
    # Step 1: Face Detection
    detection_cmd = ["python", "detection-v2.py", "--video", video_path, "--person", person_name]
    if not run_command(detection_cmd, "Face Detection"):
        print("\n❌ Pipeline aborted: Face detection failed")
        sys.exit(1)
    
    # Step 2: Model Training
    training_cmd = ["python", "train-v2.py", "--person", person_name]
    if not run_command(training_cmd, "Model Training"):
        print("\n❌ Pipeline aborted: Model training failed")
        sys.exit(1)
    
    # Step 3: Face Recognition
    if is_live_mode:
        # Use live camera for recognition
        recognition_cmd = ["python", "scan-template-v2.py", "--live"]
    else:
        # Use video file for recognition
        recognition_cmd = ["python", "scan-template-v2.py", "--video", video_path, "--person", person_name]
    
    if not run_command(recognition_cmd, "Face Recognition"):
        print("\n❌ Pipeline aborted: Face recognition failed")
        sys.exit(1)
    
    # Clean up temporary video file if in live mode
    if is_live_mode and os.path.exists(video_path):
        try:
            os.remove(video_path)
            print(f"🗑️ Temporary recording file cleaned up: {video_path}")
        except Exception as e:
            print(f"⚠️ Could not remove temporary file {video_path}: {e}")
    
    # Pipeline completed successfully
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print(f"{'='*60}")
    print(f"\n📂 All outputs saved to: {output_dir}/")
    print("\n📋 Generated files:")
    
    # List expected output files
    expected_files = [
        f"{output_dir}/{person_name}_faces_detection.json",
        f"{output_dir}/face_model.pkl"
    ]
    
    # Add recognition output files only if not in live mode
    if not is_live_mode:
        expected_files.extend([
            f"{output_dir}/recognition_output.mp4",
            f"{output_dir}/recognition_results.json"
        ])
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"   ✅ {os.path.basename(file_path)} ({file_size:,} bytes)")
        else:
            print(f"   ⚠️  {os.path.basename(file_path)} (not found)")
    
    if not is_live_mode:
        print(f"\n🎬 Recognition video: {output_dir}/recognition_output.mp4")
        print(f"📊 Results JSON: {output_dir}/recognition_results.json")
    else:
        print(f"\n📷 Live recognition completed (press 'q' to exit during recognition)")
    
    print("\n✨ Pipeline execution completed!")

if __name__ == "__main__":
    main()