import subprocess
import sys
import os
import argparse
import cv2
import time
from datetime import datetime

def record_video_from_camera(output_path, duration=10):
    """
    Record video from camera for specified duration
    
    Args:
        output_path: Path to save the recorded video
        duration: Recording duration in seconds
    """
    print(f"Starting camera recording for {duration} seconds...")
    print("Please position yourself in front of the camera and show different angles (frontal and profile views)")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    # Get camera properties
    fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    start_time = time.time()
    frame_count = 0
    
    print("Recording started! Show different face angles for better training...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        remaining = duration - elapsed
        
        if remaining <= 0:
            break
        
        # Add countdown overlay
        countdown_text = f"Recording: {remaining:.1f}s remaining"
        cv2.putText(frame, countdown_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add instruction overlay
        instruction_text = "Show frontal and profile views"
        cv2.putText(frame, instruction_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Write frame
        out.write(frame)
        
        # Show preview
        cv2.imshow('Recording - Enhanced Training Data', frame)
        
        frame_count += 1
        
        # Check for early exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user")
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Recording completed! Saved {frame_count} frames to {output_path}")
    return True

def run_detection(video_path, person_name):
    """
    Run face detection script
    
    Args:
        video_path: Path to input video
        person_name: Name of the person
    """
    print("\n=== Step 1: Face Detection ===")
    cmd = [sys.executable, "detection-v2.py", "--video", video_path, "--person", person_name]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Face detection completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Face detection failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_enhanced_training(person_name):
    """
    Run enhanced face training script
    
    Args:
        person_name: Name of the person
    """
    print("\n=== Step 2: Enhanced Face Training ===")
    cmd = [sys.executable, "train-enhanced.py", "--person", person_name]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Enhanced face training completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Enhanced face training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_enhanced_scanning(video_path, person_name, live_mode=False):
    """
    Run enhanced face scanning script
    
    Args:
        video_path: Path to input video (ignored in live mode)
        person_name: Name of the person
        live_mode: Whether to use live camera mode
    """
    print("\n=== Step 3: Enhanced Face Recognition ===")
    
    if live_mode:
        print("Starting enhanced live camera recognition...")
        print("Features enabled: Multi-scale, HOG, LBP, Angle detection, Profile optimization")
        cmd = [sys.executable, "scan-enhanced.py", "--live", "--person", person_name]
    else:
        cmd = [sys.executable, "scan-enhanced.py", "--video", video_path, "--person", person_name]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Enhanced face recognition completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Enhanced face recognition failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Face Recognition Pipeline')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', help='Input video file path')
    group.add_argument('--live', action='store_true', help='Use live camera mode (record 10s video first)')
    parser.add_argument('--person', required=True, help='Person name for the pipeline')
    args = parser.parse_args()
    
    person_name = args.person
    
    # Create output directory
    output_dir = f"faces/lock_version/{person_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ENHANCED FACE RECOGNITION PIPELINE")
    print("=" * 60)
    print(f"Person: {person_name}")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nEnhancements included:")
    print("  ✓ Multi-scale feature extraction (48x48, 64x64, 80x80)")
    print("  ✓ HOG and LBP features")
    print("  ✓ Data augmentation (6x training samples)")
    print("  ✓ Face angle detection and compensation")
    print("  ✓ Profile face optimization")
    print("  ✓ Multi-model ensemble voting")
    print("  ✓ Adaptive thresholds for different angles")
    print("=" * 60)
    
    # Determine video path
    if args.live:
        # Live mode: record video first
        video_path = os.path.join(output_dir, f"{person_name}_recorded.mp4")
        
        print("\n=== Live Mode: Recording Training Video ===")
        if not record_video_from_camera(video_path, duration=10):
            print("Failed to record video from camera!")
            return
    else:
        # Video file mode
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found!")
            return
    
    print(f"\nUsing video: {video_path}")
    
    # Step 1: Face Detection
    if not run_detection(video_path, person_name):
        print("Pipeline failed at detection step!")
        return
    
    # Step 2: Enhanced Training
    if not run_enhanced_training(person_name):
        print("Pipeline failed at enhanced training step!")
        return
    
    # Step 3: Enhanced Scanning
    if args.live:
        # In live mode, use camera for recognition
        print("\nSwitching to live camera recognition...")
        print("The enhanced model will now recognize you in real-time!")
        input("Press Enter to start live recognition (or Ctrl+C to skip)...")
        
        try:
            run_enhanced_scanning(video_path, person_name, live_mode=True)
        except KeyboardInterrupt:
            print("\nLive recognition skipped by user.")
    else:
        # Process the same video file
        if not run_enhanced_scanning(video_path, person_name, live_mode=False):
            print("Pipeline failed at enhanced scanning step!")
            return
    
    print("\n" + "=" * 60)
    print("ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nOutput files in {output_dir}:")
    print(f"  • {person_name}_faces_detection.json - Detection results")
    print(f"  • face_model_enhanced.pkl - Enhanced trained model")
    print(f"  • face_*.jpg - Detected face images")
    
    if args.live:
        print(f"  • {person_name}_recorded.mp4 - Recorded training video")
    else:
        print(f"  • recognition_output_enhanced.mp4 - Recognition results video")
        print(f"  • recognition_results_enhanced.json - Recognition results data")
    
    print("\nEnhanced features summary:")
    print("  ✓ Improved profile face recognition")
    print("  ✓ Multi-scale feature analysis")
    print("  ✓ Advanced similarity calculations")
    print("  ✓ Angle-aware processing")
    print("  ✓ Data augmentation for robust training")
    
    print("\nNext steps:")
    if args.live:
        print("  • Run with --live again to test real-time recognition")
        print("  • Use the recorded video with --video for batch processing")
    else:
        print("  • Use --live mode to test real-time recognition")
        print("  • Check the output video and JSON for detailed results")
    
    print("\nThe enhanced model should show significantly better performance")
    print("for profile faces and varying lighting conditions!")

if __name__ == "__main__":
    main()