import cv2
import os
import numpy as np
import pickle
import json
import argparse
from datetime import datetime
import glob

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

def load_dual_pca_models(dark_model_path, light_model_path):
    """
    Load both dark and light PCA models
    
    Args:
        dark_model_path (str): Path to the dark version PCA model
        light_model_path (str): Path to the light version PCA model
    
    Returns:
        tuple: (dark_model_data, light_model_data)
    """
    print("=== Loading Dual PCA Models ===")
    
    dark_model = load_pca_model(dark_model_path)
    light_model = load_pca_model(light_model_path)
    
    if dark_model is None or light_model is None:
        print("Error: Failed to load one or both models")
        return None, None
    
    print("Both models loaded successfully!")
    return dark_model, light_model

def load_face_templates(faces_dir="faces"):
    """
    Load face templates from the faces directory
    
    Args:
        faces_dir (str): Path to the faces directory containing template images
    
    Returns:
        list: List of template images (grayscale)
    """
    templates = []
    template_info = []
    
    # Get all subdirectories in faces directory
    subdirs = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
    
    print(f"Loading face templates from: {faces_dir}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(faces_dir, subdir)
        # Get all .jpg files in the subdirectory
        jpg_files = glob.glob(os.path.join(subdir_path, "*.jpg"))
        
        print(f"Found {len(jpg_files)} template images in {subdir}")
        
        for jpg_file in jpg_files[:10]:  # Limit to first 10 templates per subdirectory for performance
            try:
                # Load template image
                template_img = cv2.imread(jpg_file, cv2.IMREAD_GRAYSCALE)
                if template_img is not None:
                    templates.append(template_img)
                    template_info.append({
                        'path': jpg_file,
                        'subdir': subdir,
                        'filename': os.path.basename(jpg_file)
                    })
            except Exception as e:
                print(f"Error loading template {jpg_file}: {str(e)}")
    
    print(f"Successfully loaded {len(templates)} face templates")
    return templates, template_info

def detect_faces_with_template(gray_frame, templates, template_info, threshold=0.7, nms_threshold=0.3):
    """
    Detect faces using template matching
    
    Args:
        gray_frame (np.array): Grayscale input frame
        templates (list): List of template images
        template_info (list): Information about each template
        threshold (float): Matching threshold
        nms_threshold (float): Non-maximum suppression threshold
    
    Returns:
        list: List of detected face rectangles [(x, y, w, h)]
    """
    detections = []
    
    # Try multiple scales for template matching
    scales = [0.5, 0.7, 1.0, 1.3, 1.6]
    
    for scale in scales:
        # Resize frame for current scale
        if scale != 1.0:
            scaled_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale)
        else:
            scaled_frame = gray_frame
        
        for i, template in enumerate(templates):
            # Skip if template is larger than the scaled frame
            if template.shape[0] > scaled_frame.shape[0] or template.shape[1] > scaled_frame.shape[1]:
                continue
            
            # Perform template matching
            result = cv2.matchTemplate(scaled_frame, template, cv2.TM_CCOEFF)
            
            # Find locations where matching exceeds threshold
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                # Calculate bounding box
                x = int(pt[0] / scale)
                y = int(pt[1] / scale)
                w = int(template.shape[1] / scale)
                h = int(template.shape[0] / scale)
                
                # Store detection with confidence score
                confidence = result[int(pt[1])][int(pt[0])]
                detections.append((x, y, w, h, confidence))
    
    # Apply Non-Maximum Suppression to remove overlapping detections
    if len(detections) > 0:
        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes = [(x, y, w, h) for x, y, w, h, _ in detections]
        scores = [conf for _, _, _, _, conf in detections]
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, nms_threshold)
        
        # Filter detections based on NMS results
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_detections = [(detections[i][0], detections[i][1], detections[i][2], detections[i][3]) 
                                 for i in indices]
        else:
            filtered_detections = []
    else:
        filtered_detections = []
    
    return filtered_detections

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

def recognize_face_dual_model(face_vector, dark_model_data, light_model_data, similarity_threshold=0.7):
    """
    Recognize a face using both dark and light PCA models (OR logic)
    
    Args:
        face_vector (np.array): Flattened face image to recognize
        dark_model_data (dict): Dark version PCA model data
        light_model_data (dict): Light version PCA model data
        similarity_threshold (float): Minimum similarity for recognition
    
    Returns:
        tuple: (person_name, best_confidence, is_recognized, dark_similarity, light_similarity)
    """
    # Test with dark model
    dark_name, dark_similarity, dark_recognized = recognize_face(
        face_vector, dark_model_data, similarity_threshold
    )
    
    # Test with light model
    light_name, light_similarity, light_recognized = recognize_face(
        face_vector, light_model_data, similarity_threshold
    )
    
    # OR logic: recognized if either model recognizes the face
    is_recognized = dark_recognized or light_recognized
    
    # Use the higher similarity as the best confidence
    best_confidence = max(dark_similarity, light_similarity)
    
    # Use the person name from the model with higher similarity
    person_name = dark_name if dark_similarity >= light_similarity else light_name
    
    return person_name, best_confidence, is_recognized, dark_similarity, light_similarity

def detect_and_recognize_faces(frame, templates, template_info, model_data, similarity_threshold=0.7, template_threshold=0.7):
    """
    Detect and recognize faces in a frame using template matching (single model version)
    
    Args:
        frame (np.array): Input video frame
        templates (list): List of face template images
        template_info (list): Information about each template
        model_data (dict): Loaded PCA model data
        similarity_threshold (float): Recognition threshold
        template_threshold (float): Template matching threshold
    
    Returns:
        list: List of detection results [(x, y, w, h, name, confidence, recognized)]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using template matching
    faces = detect_faces_with_template(gray, templates, template_info, template_threshold)
    
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

def detect_and_recognize_faces_dual_model(frame, templates, template_info, dark_model_data, light_model_data, similarity_threshold=0.7, template_threshold=0.7):
    """
    Detect and recognize faces in a frame using template matching and dual models (dark + light)
    
    Args:
        frame (np.array): Input video frame
        templates (list): List of face template images
        template_info (list): Information about each template
        dark_model_data (dict): Dark version PCA model data
        light_model_data (dict): Light version PCA model data
        similarity_threshold (float): Recognition threshold
        template_threshold (float): Template matching threshold
    
    Returns:
        list: List of detection results [(x, y, w, h, name, confidence, recognized)]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using template matching
    faces = detect_faces_with_template(gray, templates, template_info, template_threshold)
    
    results = []
    
    for i, (x, y, w, h) in enumerate(faces):
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize face to match training data dimensions
        face_dim = int(np.sqrt(dark_model_data['face_dimensions']))
        face_resized = cv2.resize(face_roi, (face_dim, face_dim))
        
        # Flatten face to vector
        face_vector = face_resized.flatten().astype(np.float64)
        
        # Recognize face using dual models
        person_name, best_confidence, is_recognized, dark_similarity, light_similarity = recognize_face_dual_model(
            face_vector, dark_model_data, light_model_data, similarity_threshold
        )
        
        # Print detailed similarity values for debugging
        status = "RECOGNIZED" if is_recognized else "UNKNOWN"
        print(f"Face {i+1} at position ({x}, {y}, {w}x{h}):")
        print(f"  Dark model:  similarity={dark_similarity:.4f}, threshold={similarity_threshold:.4f}")
        print(f"  Light model: similarity={light_similarity:.4f}, threshold={similarity_threshold:.4f}")
        print(f"  Best confidence: {best_confidence:.4f}, Status: {status}")
        
        results.append((x, y, w, h, person_name, best_confidence, is_recognized))
    
    return results

def draw_face_annotations(frame, detection_results, avg_face_size=None):
    """
    Draw bounding boxes and labels on detected faces
    
    Args:
        frame (np.array): Input video frame
        detection_results (list): List of detection results
        avg_face_size (float): Average face size for dynamic filtering
    
    Returns:
        np.array: Annotated frame
    """
    annotated_frame = frame.copy()
    
    for (x, y, w, h, person_name, confidence, is_recognized) in detection_results:
        # Skip detection results with confidence < 0.3 and not recognized
        if confidence < 0.3 and not is_recognized:
            continue
            
        # Dynamic face size filtering based on average size
        if avg_face_size is not None:
            current_face_size = (w + h) / 2
            if current_face_size < avg_face_size * 0.5:  # Skip faces smaller than 50% of average
                continue
        else:
            # Fallback to original size filtering
            if (w + h) / 2 < 200:
                continue
            
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

def process_video(input_video_path, dark_model_path, light_model_path, output_video_path, similarity_threshold=0.7, template_threshold=0.7):
    """
    Process video to detect and recognize faces using template matching and dual models
    
    Args:
        input_video_path (str): Path to input video
        dark_model_path (str): Path to dark PCA model file
        light_model_path (str): Path to light PCA model file
        output_video_path (str): Path to save output video
        similarity_threshold (float): Recognition threshold
        template_threshold (float): Template matching threshold
    
    Returns:
        bool: Success status
    """
    # Load dual PCA models
    dark_model_data, light_model_data = load_dual_pca_models(dark_model_path, light_model_path)
    if dark_model_data is None or light_model_data is None:
        return False
    
    # Load face templates
    templates, template_info = load_face_templates()
    if len(templates) == 0:
        print("Error: No face templates loaded")
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
    print(f"Using {len(templates)} face templates for detection")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # First pass: collect face sizes for dynamic filtering
    print("First pass: collecting face sizes...")
    face_sizes = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:  # Sample every 10th frame for efficiency
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces_with_template(gray, templates, template_info, template_threshold)
            
            for (x, y, w, h) in faces:
                face_sizes.append((w + h) / 2)
    
    # Calculate average face size
    avg_face_size = np.mean(face_sizes) if face_sizes else None
    print(f"Average face size: {avg_face_size:.2f}" if avg_face_size else "No faces detected in sampling")
    
    # Reset video capture for second pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Second pass: process frames with dynamic filtering
    print("Second pass: processing frames...")
    frame_count = 0
    recognition_stats = {'recognized': 0, 'unknown': 0, 'total_detections': 0}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect and recognize faces using template matching and dual models
        detection_results = detect_and_recognize_faces_dual_model(
            frame, templates, template_info, dark_model_data, light_model_data, 
            similarity_threshold, template_threshold
        )
        
        # Update statistics
        for (_, _, _, _, _, _, is_recognized) in detection_results:
            recognition_stats['total_detections'] += 1
            if is_recognized:
                recognition_stats['recognized'] += 1
            else:
                recognition_stats['unknown'] += 1
        
        # Draw annotations with dynamic filtering
        annotated_frame = draw_face_annotations(frame, detection_results, avg_face_size)
        
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

def process_live_camera(dark_model_path, light_model_path, similarity_threshold=0.7, template_threshold=0.7):
    """
    Process live camera feed to detect and recognize faces in real-time using template matching and dual models
    
    Args:
        dark_model_path (str): Path to dark PCA model file
        light_model_path (str): Path to light PCA model file
        similarity_threshold (float): Recognition threshold
        template_threshold (float): Template matching threshold
    
    Returns:
        bool: Success status
    """
    # Load dual PCA models
    dark_model_data, light_model_data = load_dual_pca_models(dark_model_path, light_model_path)
    if dark_model_data is None or light_model_data is None:
        return False
    
    # Load face templates
    templates, template_info = load_face_templates()
    if len(templates) == 0:
        print("Error: No face templates loaded")
        return False
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return False
    
    print("Live camera face recognition started. Press 'q' to quit.")
    print(f"Using {len(templates)} face templates for detection")
    
    # Initialize face size history for dynamic filtering
    face_size_history = []
    max_history_size = 50
    
    # Process frames in real-time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Detect and recognize faces using template matching and dual models
        detection_results = detect_and_recognize_faces_dual_model(
            frame, templates, template_info, dark_model_data, light_model_data, 
            similarity_threshold, template_threshold
        )
        
        # Update face size history for dynamic filtering
        for (x, y, w, h, _, _, _) in detection_results:
            face_size_history.append((w + h) / 2)
        
        # Maintain history size limit
        if len(face_size_history) > max_history_size:
            face_size_history = face_size_history[-max_history_size:]
        
        # Calculate dynamic average face size
        avg_face_size = np.mean(face_size_history) if face_size_history else None
        
        # Draw annotations with dynamic filtering
        annotated_frame = draw_face_annotations(frame, detection_results, avg_face_size)
        
        # Display the frame
        cv2.imshow('Live Face Recognition (Template Matching) - Press q to quit', annotated_frame)
        
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
    parser = argparse.ArgumentParser(description='Face Detection and Recognition System (Template Matching)')
    parser.add_argument('--video', type=str, help='Path to input video file for processing')
    parser.add_argument('--live', action='store_true', help='Use live camera for real-time face recognition')
    parser.add_argument('--template-threshold', type=float, default=0.7, help='Template matching threshold (default: 0.7)')
    
    args = parser.parse_args()
    
    # Configuration - Dual model paths
    dark_model_file = "models/Joseph_Lai_dark_pca_model.pkl"
    light_model_file = "models/Joseph_Lai_light_pca_model.pkl"
    similarity_threshold = 0.8
    template_threshold = args.template_threshold
    
    try:
        # Check if both model files exist
        if not os.path.exists(dark_model_file):
            raise FileNotFoundError(f"Dark PCA model not found: {dark_model_file}")
        if not os.path.exists(light_model_file):
            raise FileNotFoundError(f"Light PCA model not found: {light_model_file}")
        
        if args.live:
            # Live camera mode
            print("Starting live camera mode with template matching...")
            success = process_live_camera(dark_model_file, light_model_file, similarity_threshold, template_threshold)
            
        elif args.video:
            # Video file mode
            input_video = args.video
            output_dir = "output"
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output video filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = os.path.splitext(os.path.basename(input_video))[0]
            output_video = os.path.join(output_dir, f"recognized_template_{video_name}_{timestamp}.mp4")
            
            # Check if input video exists
            if not os.path.exists(input_video):
                raise FileNotFoundError(f"Input video not found: {input_video}")
            print(f"Processing video file with template matching: {input_video}")
            success = process_video(input_video, dark_model_file, light_model_file, output_video, 
                                  similarity_threshold, template_threshold)
            
            if success:
                print(f"\n=== Face Recognition Complete ===")
                print(f"Input video: {input_video}")
                print(f"Output video: {output_video}")
        
        else:
            # No arguments provided, show help
            parser.print_help()
            print("\nExamples:")
            print("  python scan-template.py --video videos/test.mp4")
            print("  python scan-template.py --live")
            print("  python scan-template.py --video videos/test.mp4 --template-threshold 0.8")
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