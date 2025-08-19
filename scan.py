import cv2
import os
import numpy as np
import pickle
import json
import argparse
from datetime import datetime

def load_pca_model(model_path):
    """
    Load the trained PCA model from file
    
    Args:
        model_path (str): Path to the saved PCA model file
    
    Returns:
        dict: Model data containing eigenfaces, mean_face, etc.
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"PCA model loaded successfully from: {model_path}")
        print(f"Person: {model_data['person_name']}")
        print(f"Training timestamp: {model_data['training_timestamp']}")
        print(f"Number of components: {model_data['n_components']}")
        print(f"Face dimensions: {model_data['face_dimensions']}")
        
        return model_data
    
    except Exception as e:
        print(f"Error loading PCA model: {str(e)}")
        return None

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors
    
    Args:
        vec1 (np.array): First vector
        vec2 (np.array): Second vector
    
    Returns:
        float: Cosine similarity value
    """
    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Calculate cosine similarity
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return similarity

def project_face_to_eigenspace(face_vector, eigenfaces, mean_face):
    """
    Project a face vector into the eigenface space
    
    Args:
        face_vector (np.array): Flattened face image
        eigenfaces (np.array): Principal component vectors
        mean_face (np.array): Mean face vector
    
    Returns:
        np.array: Projected face vector in eigenspace
    """
    # Center the face by subtracting mean
    centered_face = face_vector - mean_face
    
    # Project onto eigenfaces
    projected_face = np.dot(centered_face, eigenfaces)
    
    return projected_face

def recognize_face(face_vector, model_data, similarity_threshold=0.7):
    """
    Recognize a face using the trained PCA model
    
    Args:
        face_vector (np.array): Flattened face image to recognize
        model_data (dict): Loaded PCA model data
        similarity_threshold (float): Minimum similarity for recognition
    
    Returns:
        tuple: (person_name, confidence, is_recognized)
    """
    eigenfaces = model_data['eigenfaces']
    mean_face = model_data['mean_face']
    projected_training_data = model_data['projected_data']
    person_name = model_data['person_name']
    
    # Project the input face to eigenspace
    projected_face = project_face_to_eigenspace(face_vector, eigenfaces, mean_face)
    
    # Calculate similarities with all training faces
    similarities = []
    for training_face in projected_training_data:
        similarity = cosine_similarity(projected_face, training_face)
        similarities.append(similarity)
    
    # Get the best match
    max_similarity = max(similarities)
    
    # Check if similarity exceeds threshold
    is_recognized = max_similarity >= similarity_threshold
    
    return person_name, max_similarity, is_recognized

def detect_and_recognize_faces(frame, face_cascade, model_data, similarity_threshold=0.7):
    """
    Detect and recognize faces in a frame
    
    Args:
        frame (np.array): Input video frame
        face_cascade: OpenCV face cascade classifier
        model_data (dict): Loaded PCA model data
        similarity_threshold (float): Recognition threshold
    
    Returns:
        list: List of detection results [(x, y, w, h, name, confidence, recognized)]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    results = []
    
    for i, (x, y, w, h) in enumerate(faces):
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize face to match training data dimensions
        face_dim = int(np.sqrt(model_data['face_dimensions']))
        face_resized = cv2.resize(face_roi, (face_dim, face_dim))
        
        # Flatten face to vector
        face_vector = face_resized.flatten().astype(np.float64)
        
        # Recognize face
        person_name, confidence, is_recognized = recognize_face(
            face_vector, model_data, similarity_threshold
        )
        
        # Print similarity value for debugging
        status = "RECOGNIZED" if is_recognized else "UNKNOWN"
        print(f"Face {i+1} at position ({x}, {y}, {w}x{h}): similarity={confidence:.4f}, threshold={similarity_threshold:.4f}, status={status}")
        
        results.append((x, y, w, h, person_name, confidence, is_recognized))
    
    return results

def draw_face_annotations(frame, detection_results):
    """
    Draw bounding boxes and labels on detected faces
    
    Args:
        frame (np.array): Input video frame
        detection_results (list): List of detection results
    
    Returns:
        np.array: Annotated frame
    """
    annotated_frame = frame.copy()
    
    for (x, y, w, h, person_name, confidence, is_recognized) in detection_results:
        # All face detection boxes are RED SQUARES
        box_color = (0, 0, 255)  # Red color for all detection boxes
        
        # Make it a square by using the larger dimension
        size = max(w, h)
        # Center the square on the detected face
        square_x = x + (w - size) // 2
        square_y = y + (h - size) // 2
        
        # Draw red square bounding box
        cv2.rectangle(annotated_frame, (square_x, square_y), (square_x + size, square_y + size), box_color, 2)
        
        # Choose label color based on recognition status
        if is_recognized:
            label_color = (255, 255, 0)  # Cyan for recognized faces
            label = f"{person_name} ({confidence:.2f})"
        else:
            label_color = (0, 0, 255)  # Red for unknown faces
            label = f"Unknown ({confidence:.2f})"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(
            annotated_frame,
            (x, y - label_size[1] - 10),
            (x + label_size[0], y),
            label_color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated_frame,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return annotated_frame

def process_video(input_video_path, model_path, output_video_path, similarity_threshold=0.7):
    """
    Process video to detect and recognize faces
    
    Args:
        input_video_path (str): Path to input video
        model_path (str): Path to PCA model file
        output_video_path (str): Path to save output video
        similarity_threshold (float): Recognition threshold
    
    Returns:
        bool: Success status
    """
    # Load PCA model
    model_data = load_pca_model(model_path)
    if model_data is None:
        return False
    
    # Initialize face cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return False
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {input_video_path}")
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    recognition_stats = {'recognized': 0, 'unknown': 0, 'total_detections': 0}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect and recognize faces
        detection_results = detect_and_recognize_faces(
            frame, face_cascade, model_data, similarity_threshold
        )
        
        # Update statistics
        for (_, _, _, _, _, _, is_recognized) in detection_results:
            recognition_stats['total_detections'] += 1
            if is_recognized:
                recognition_stats['recognized'] += 1
            else:
                recognition_stats['unknown'] += 1
        
        # Draw annotations
        annotated_frame = draw_face_annotations(frame, detection_results)
        
        # Write frame to output video
        out.write(annotated_frame)
        
        # Print progress
        if frame_count % 30 == 0:  # Print every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # Clean up
    cap.release()
    out.release()
    
    # Print final statistics
    print(f"\n=== Processing Complete ===")
    print(f"Output video saved to: {output_video_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total face detections: {recognition_stats['total_detections']}")
    print(f"Recognized faces: {recognition_stats['recognized']}")
    print(f"Unknown faces: {recognition_stats['unknown']}")
    
    if recognition_stats['total_detections'] > 0:
        recognition_rate = (recognition_stats['recognized'] / recognition_stats['total_detections']) * 100
        print(f"Recognition rate: {recognition_rate:.1f}%")
    
    return True

def process_live_camera(model_path, similarity_threshold=0.7):
    """
    Process live camera feed to detect and recognize faces in real-time
    
    Args:
        model_path (str): Path to PCA model file
        similarity_threshold (float): Recognition threshold
    
    Returns:
        bool: Success status
    """
    # Load PCA model
    model_data = load_pca_model(model_path)
    if model_data is None:
        return False
    
    # Initialize face cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return False
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    print("Live camera face recognition started. Press 'q' to quit.")
    # print(f"Using similarity threshold: {similarity_threshold}")
    
    # Process frames in real-time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Detect and recognize faces
        detection_results = detect_and_recognize_faces(
            frame, face_cascade, model_data, similarity_threshold
        )
        
        # Draw annotations
        annotated_frame = draw_face_annotations(frame, detection_results)
        
        # Display the frame
        cv2.imshow('Live Face Recognition - Press q to quit', annotated_frame)
        
        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    print("Live camera face recognition stopped.")
    return True

def main():
    """
    Main function with command line argument support
    """
    parser = argparse.ArgumentParser(description='Face Detection and Recognition System')
    parser.add_argument('--video', type=str, help='Path to input video file for processing')
    parser.add_argument('--live', action='store_true', help='Use live camera for real-time face recognition')
    
    args = parser.parse_args()
    
    # Configuration
    model_file = "models/Joseph_Lai_pca_model.pkl"
    similarity_threshold = 0.9  # Adjust based on requirements
    
    try:
        # Check if model file exists
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"PCA model not found: {model_file}")
        
        if args.live:
            # Live camera mode
            print("Starting live camera mode...")
            success = process_live_camera(model_file, similarity_threshold)
            
        elif args.video:
            # Video file mode
            input_video = args.video
            output_dir = "output"
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output video filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = os.path.splitext(os.path.basename(input_video))[0]
            output_video = os.path.join(output_dir, f"recognized_{video_name}_{timestamp}.mp4")
            
            # Check if input video exists
            if not os.path.exists(input_video):
                raise FileNotFoundError(f"Input video not found: {input_video}")
            
            print(f"Processing video file: {input_video}")
            success = process_video(input_video, model_file, output_video, similarity_threshold)
            
            if success:
                print(f"\n=== Face Recognition Complete ===")
                print(f"Input video: {input_video}")
                print(f"Output video: {output_video}")
                # print(f"Similarity threshold: {similarity_threshold}")
        
        else:
            # No arguments provided, show help
            parser.print_help()
            print("\nExamples:")
            print("  python scan.py --video videos/test.mp4")
            print("  python scan.py --live")
            return False
        
        if not success:
            print("Face recognition failed")
            return False
    
    except Exception as e:
        print(f"Error during face recognition: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    main()