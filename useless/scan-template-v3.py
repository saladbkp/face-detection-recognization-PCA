import cv2
import numpy as np
import pickle
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class FaceScanner:
    def __init__(self):
        """
        Initialize face scanner for recognition using multiple trained PCA models
        """
        self.models = {}  # Dictionary to store multiple models
        self.face_shape = (64, 64)
        self.is_loaded = False
        
        # Face detection setup
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Recognition parameters
        self.similarity_threshold = 0.7  # Cosine similarity threshold
        self.confidence_threshold = 0.6  # Recognition confidence threshold
        
    def load_all_models(self, models_base_path="faces/lock_version"):
        """
        Load all trained models from the base directory
        
        Args:
            models_base_path: Base path containing person directories with face_model.pkl files
        """
        if not os.path.exists(models_base_path):
            print(f"Error: Models base path {models_base_path} not found!")
            return False
        
        print(f"Loading all models from {models_base_path}")
        
        loaded_count = 0
        total_faces = 0
        
        # Scan for person directories
        for person_name in os.listdir(models_base_path):
            person_dir = os.path.join(models_base_path, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            model_path = os.path.join(person_dir, "face_model.pkl")
            if not os.path.exists(model_path):
                print(f"Warning: No model file found for {person_name}")
                continue
            
            try:
                print(f"Loading model for {person_name}...")
                
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Store model data for this person
                self.models[person_name] = {
                    'pca': model_data['pca'],
                    'scaler': model_data['scaler'],
                    'face_features': model_data['face_features'],
                    'face_labels': model_data['face_labels'],
                    'face_info': model_data['face_info'],
                    'person_id_map': model_data['person_id_map'],
                    'mean_face': model_data.get('mean_face', None),
                    'eigenfaces': model_data.get('eigenfaces', None),
                    'face_shape': model_data.get('face_shape', (64, 64)),
                    'training_date': model_data.get('training_date', 'Unknown')
                }
                
                loaded_count += 1
                total_faces += len(model_data['face_features'])
                
                print(f"  - {person_name}: {len(model_data['face_features'])} faces")
                
            except Exception as e:
                print(f"Error loading model for {person_name}: {e}")
                continue
        
        if loaded_count > 0:
            self.is_loaded = True
            print(f"\nSuccessfully loaded {loaded_count} models with {total_faces} total faces")
            print(f"Available persons: {list(self.models.keys())}")
            return True
        else:
            print("No models loaded!")
            return False
    
    def extract_face_features_for_person(self, face_image, person_name):
        """
        Extract features from a face image using a specific person's PCA model
        
        Args:
            face_image: Grayscale face image (numpy array)
            person_name: Name of the person whose model to use
            
        Returns:
            Feature vector or None if extraction fails
        """
        if not self.is_loaded or person_name not in self.models:
            return None
        
        try:
            model = self.models[person_name]
            
            # Resize to standard size
            resized = cv2.resize(face_image, model['face_shape'])
            
            # Flatten to 1D vector
            face_vector = resized.flatten().reshape(1, -1)
            
            # Apply same preprocessing as training
            scaled_features = model['scaler'].transform(face_vector)
            
            # Apply PCA transformation
            pca_features = model['pca'].transform(scaled_features)
            
            return pca_features[0]
        
        except Exception as e:
            print(f"Error extracting features for {person_name}: {e}")
            return None
    
    def recognize_face_with_all_models(self, face_image):
        """
        Recognize a face using all loaded models and return the best match
        
        Args:
            face_image: Grayscale face image (numpy array)
            
        Returns:
            Dictionary with recognition results
        """
        if not self.is_loaded:
            return {'recognized': False, 'person_name': 'Unknown', 'confidence': 0.0}
        
        best_result = {'recognized': False, 'person_name': 'Unknown', 'confidence': 0.0}
        
        # Test against each person's model
        for person_name, model in self.models.items():
            try:
                # Extract features using this person's model
                face_features = self.extract_face_features_for_person(face_image, person_name)
                if face_features is None:
                    continue
                
                # Calculate cosine similarity with this person's stored faces
                similarities = cosine_similarity([face_features], model['face_features'])[0]
                
                # Find best match for this person
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                
                # Check if this is better than current best
                if best_similarity > best_result['confidence']:
                    best_result = {
                        'recognized': best_similarity >= self.similarity_threshold,
                        'person_name': person_name if best_similarity >= self.similarity_threshold else 'Unknown',
                        'confidence': float(best_similarity),
                        'match_index': int(best_match_idx)
                    }
                
            except Exception as e:
                print(f"Error testing against {person_name} model: {e}")
                continue
        
        return best_result
    
    def process_live_camera(self):
        """
        Process live camera feed for real-time face recognition
        """
        if not self.is_loaded:
            print("Error: Model not loaded!")
            return
        
        print("Starting live camera face recognition...")
        print("Press 'q' to quit")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        recognition_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each detected face
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Recognize face using all models
                result = self.recognize_face_with_all_models(face_roi)
                
                # Draw rectangle around face
                color = (0, 255, 0) if result['recognized'] else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Prepare label text
                if result['recognized']:
                    label = f"{result['person_name']} ({result['confidence']:.2f})"
                else:
                    label = f"Unknown ({result['confidence']:.2f})"
                
                # Draw label
                label_y = y - 10 if y - 10 > 10 else y + h + 20
                cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Store result for logging
                if frame_count % 30 == 0:  # Log every 30 frames
                    recognition_results.append({
                        'timestamp': datetime.now().isoformat(),
                        'frame': frame_count,
                        'face_id': i,
                        'recognized': result['recognized'],
                        'person_name': result['person_name'],
                        'confidence': result['confidence']
                    })
            
            # Add frame info
            info_text = f"Frame: {frame_count} | Faces: {len(faces)} | Press 'q' to quit"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Face Recognition - Live Camera', frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nLive recognition completed!")
        print(f"Total frames processed: {frame_count}")
        print(f"Recognition events logged: {len(recognition_results)}")
        
        return recognition_results
    
    def process_video_file(self, video_path, output_path=None):
        """
        Process video file for face recognition
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
        """
        if not self.is_loaded:
            print("Error: Model not loaded!")
            return
        
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found!")
            return
        
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        recognition_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each detected face
            for i, (x, y, w, h) in enumerate(faces):
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Recognize face using all models
                result = self.recognize_face_with_all_models(face_roi)
                
                # Draw rectangle around face
                color = (0, 255, 0) if result['recognized'] else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Prepare label text
                if result['recognized']:
                    label = f"{result['person_name']} ({result['confidence']:.2f})"
                else:
                    label = f"Unknown ({result['confidence']:.2f})"
                
                # Draw label
                label_y = y - 10 if y - 10 > 10 else y + h + 20
                cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Store result
                recognition_results.append({
                    'timestamp': frame_count / fps,
                    'frame': frame_count,
                    'face_id': i,
                    'recognized': result['recognized'],
                    'person_name': result['person_name'],
                    'confidence': result['confidence']
                })
            
            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Faces: {len(faces)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame to output video
            if out:
                out.write(frame)
            
            # Show progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        
        print(f"\nVideo processing completed!")
        print(f"Total frames processed: {frame_count}")
        print(f"Recognition events: {len(recognition_results)}")
        if output_path:
            print(f"Output video saved to: {output_path}")
        
        return recognition_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face recognition using all trained PCA models')
    parser.add_argument('--live', action='store_true', help='Use live camera for recognition')
    parser.add_argument('--video', help='Video file path for recognition')
    parser.add_argument('--output', help='Output video path (for video processing)')
    args = parser.parse_args()
    
    # Create scanner and load all models
    scanner = FaceScanner()
    if not scanner.load_all_models():
        print("Failed to load models!")
        print("Please make sure you have trained models in faces/lock_version/")
        print("Run: python train-v3.py --person <person_name> to train models")
        return
    
    # Process based on arguments
    if args.live:
        print("\nStarting live recognition with all models...")
        results = scanner.process_live_camera()
        
    elif args.video:
        print("\nProcessing video with all models...")
        results = scanner.process_video_file(args.video, args.output)
        
        # Print summary
        if results:
            recognized_count = sum(1 for r in results if r['recognized'])
            person_counts = {}
            for r in results:
                if r['recognized']:
                    person_name = r['person_name']
                    person_counts[person_name] = person_counts.get(person_name, 0) + 1
            
            print(f"\nRecognition Summary:")
            print(f"Total detections: {len(results)}")
            print(f"Recognized faces: {recognized_count}")
            print(f"Unknown faces: {len(results) - recognized_count}")
            if person_counts:
                print("Recognition breakdown:")
                for person, count in person_counts.items():
                    print(f"  - {person}: {count} detections")
    
    else:
        print("Please specify --live for camera or --video <path> for video file")
        print("Example: python scan-template-v3.py --live")
        print("Example: python scan-template-v3.py --video test.mp4")

if __name__ == "__main__":
    main()