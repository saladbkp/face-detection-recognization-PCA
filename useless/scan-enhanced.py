import cv2
import json
import os
import numpy as np
import pickle
import argparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedFaceScanner:
    def __init__(self):
        """
        Initialize enhanced face scanner with multi-model support
        """
        self.trained_models = {}
        self.face_info = []
        self.person_id_map = {}
        self.is_loaded = False
        self.detection_data = None
        
        # Enhanced recognition parameters
        self.base_threshold = 0.6
        self.profile_threshold = 0.5  # Lower threshold for profile faces
        self.confidence_weights = {
            'scale_48': 0.15,
            'scale_64': 0.25,
            'scale_80': 0.20,
            'hog': 0.25,
            'lbp': 0.15
        }
        
        # Face angle detector
        self.angle_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Feature cache for performance
        self.feature_cache = {}
    
    def detect_face_angle(self, face_img):
        """
        Detect if face is frontal or profile (side view)
        
        Args:
            face_img: Face image in grayscale
            
        Returns:
            angle_type: 'frontal', 'left_profile', 'right_profile'
        """
        # Detect profile faces
        profiles = self.angle_detector.detectMultiScale(face_img, 1.1, 4)
        
        if len(profiles) > 0:
            # Check which side the profile is on
            h, w = face_img.shape
            center_x = w // 2
            
            for (x, y, w_p, h_p) in profiles:
                profile_center = x + w_p // 2
                if profile_center < center_x:
                    return 'left_profile'
                else:
                    return 'right_profile'
        
        return 'frontal'
    
    def extract_hog_features(self, face_img):
        """
        Extract HOG (Histogram of Oriented Gradients) features
        
        Args:
            face_img: Face image in grayscale
            
        Returns:
            HOG features vector
        """
        features = hog(face_img, 
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys',
                      visualize=False,
                      feature_vector=True)
        return features
    
    def extract_lbp_features(self, face_img):
        """
        Extract LBP (Local Binary Pattern) features
        
        Args:
            face_img: Face image in grayscale
            
        Returns:
            LBP features histogram
        """
        radius = 3
        n_points = 8 * radius
        
        lbp = local_binary_pattern(face_img, n_points, radius, method='uniform')
        
        # Calculate histogram
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist
    
    def extract_multiscale_features(self, face_img, angle_type='frontal'):
        """
        Extract features at multiple scales and using different methods
        
        Args:
            face_img: Face image in grayscale
            angle_type: Type of face angle ('frontal', 'left_profile', 'right_profile')
            
        Returns:
            Dictionary of feature vectors
        """
        # Check cache first
        cache_key = f"{hash(face_img.tobytes())}_{angle_type}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        features = {}
        
        # Multi-scale pixel features
        scales = [48, 64, 80]
        for scale in scales:
            resized = cv2.resize(face_img, (scale, scale))
            
            # Apply angle-specific preprocessing
            if angle_type in ['left_profile', 'right_profile']:
                # Enhance contrast for profile faces
                resized = cv2.equalizeHist(resized)
                
                # Apply edge enhancement
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                resized = cv2.filter2D(resized, -1, kernel)
                resized = np.clip(resized, 0, 255).astype(np.uint8)
            
            features[f'scale_{scale}'] = resized.flatten()
        
        # HOG features
        hog_features = self.extract_hog_features(cv2.resize(face_img, (64, 64)))
        features['hog'] = hog_features
        
        # LBP features
        lbp_features = self.extract_lbp_features(cv2.resize(face_img, (64, 64)))
        features['lbp'] = lbp_features
        
        # Cache the result
        self.feature_cache[cache_key] = features
        
        return features
    
    def load_enhanced_model(self, model_path):
        """
        Load enhanced trained model
        
        Args:
            model_path: Path to the enhanced model file
        """
        if not os.path.exists(model_path):
            print(f"Error: Enhanced model file {model_path} not found!")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if it's an enhanced model
            if model_data.get('model_type') != 'enhanced':
                print(f"Warning: Model {model_path} is not an enhanced model!")
                return False
            
            self.trained_models = model_data['trained_models']
            self.face_info = model_data['face_info']
            self.person_id_map = model_data['person_id_map']
            self.is_loaded = True
            
            print(f"Enhanced model loaded from {model_path}")
            print(f"Training date: {model_data.get('training_date', 'Unknown')}")
            print(f"Total faces: {len(self.face_info)}")
            print(f"Total persons: {len(self.person_id_map)}")
            print(f"Available feature types: {list(self.trained_models.keys())}")
            
            return True
            
        except Exception as e:
            print(f"Error loading enhanced model: {e}")
            return False
    
    def load_detection_data(self, json_path):
        """
        Load face detection data from JSON file
        
        Args:
            json_path: Path to detection JSON file
        """
        if not os.path.exists(json_path):
            print(f"Error: Detection data file {json_path} not found!")
            return False
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.detection_data = json.load(f)
            
            print(f"Detection data loaded: {len(self.detection_data['faces'])} faces")
            return True
            
        except Exception as e:
            print(f"Error loading detection data: {e}")
            return False
    
    def get_reference_position(self):
        """
        Get reference position from detection data
        
        Returns:
            Reference position (x, y, w, h) or None
        """
        if not self.detection_data or len(self.detection_data['faces']) == 0:
            return None
        
        first_face = self.detection_data['faces'][0]
        return (first_face['x'], first_face['y'], first_face['width'], first_face['height'])
    
    def template_match_in_region(self, frame, template_img, search_region, threshold=0.6):
        """
        Perform template matching in a specific region
        
        Args:
            frame: Current video frame
            template_img: Template image for matching
            search_region: (x, y, w, h) region to search in
            threshold: Matching threshold
            
        Returns:
            Best match result or None
        """
        x, y, w, h = search_region
        
        # Extract search region from frame
        search_area = frame[y:y+h, x:x+w]
        
        if search_area.size == 0:
            return None
        
        # Perform template matching
        result = cv2.matchTemplate(search_area, template_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            # Convert coordinates back to full frame
            match_x = x + max_loc[0]
            match_y = y + max_loc[1]
            match_w, match_h = template_img.shape[1], template_img.shape[0]
            
            return {
                'x': match_x,
                'y': match_y,
                'w': match_w,
                'h': match_h,
                'confidence': max_val
            }
        
        return None
    
    def enhanced_recognize_face(self, face_img):
        """
        Enhanced face recognition using multiple models and features
        
        Args:
            face_img: Face image in grayscale
            
        Returns:
            (person_id, person_name, confidence, angle_type)
        """
        if not self.is_loaded:
            return -1, "unknown", 0.0, "unknown"
        
        # Detect face angle
        angle_type = self.detect_face_angle(face_img)
        
        # Extract multi-scale features
        input_features = self.extract_multiscale_features(face_img, angle_type)
        
        # Calculate similarities for each feature type
        similarities = {}
        confidences = {}
        
        for feature_type, input_feature in input_features.items():
            if feature_type not in self.trained_models:
                continue
            
            model_data = self.trained_models[feature_type]
            stored_features = model_data['features']
            scaler = model_data['scaler']
            pca = model_data['pca']
            
            # Transform input feature
            try:
                input_feature_scaled = scaler.transform([input_feature])
                input_feature_pca = pca.transform(input_feature_scaled)
                
                # Calculate cosine similarity
                cos_sim = cosine_similarity(input_feature_pca, stored_features)[0]
                
                # Calculate euclidean distance (convert to similarity)
                euc_dist = euclidean_distances(input_feature_pca, stored_features)[0]
                euc_sim = 1 / (1 + euc_dist)  # Convert distance to similarity
                
                # Combine similarities
                combined_sim = 0.7 * cos_sim + 0.3 * euc_sim
                
                similarities[feature_type] = combined_sim
                confidences[feature_type] = np.max(combined_sim)
                
            except Exception as e:
                print(f"Error processing {feature_type} features: {e}")
                continue
        
        if not similarities:
            return -1, "unknown", 0.0, angle_type
        
        # Multi-model voting and confidence calculation
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for feature_type, confidence in confidences.items():
            weight = self.confidence_weights.get(feature_type, 0.2)
            
            # Apply angle-specific adjustments
            if angle_type in ['left_profile', 'right_profile']:
                # Boost HOG and LBP for profile faces
                if feature_type in ['hog', 'lbp']:
                    weight *= 1.3
                # Reduce pixel-based features for profile faces
                elif feature_type.startswith('scale_'):
                    weight *= 0.8
            
            weighted_confidence += confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            final_confidence = weighted_confidence / total_weight
        else:
            final_confidence = 0.0
        
        # Apply angle-specific threshold
        threshold = self.profile_threshold if angle_type in ['left_profile', 'right_profile'] else self.base_threshold
        
        # Additional boost for profile faces if confidence is close
        if angle_type in ['left_profile', 'right_profile'] and final_confidence > 0.4:
            final_confidence *= 1.2  # 20% boost for profile faces
        
        # Determine recognition result
        if final_confidence >= threshold:
            # Find the person with highest average similarity
            best_person_id = 0  # Since we only have one person in training
            person_name = list(self.person_id_map.keys())[0]
            return best_person_id, person_name, final_confidence, angle_type
        else:
            return -1, "unknown", final_confidence, angle_type
    
    def process_video(self, video_path, output_video_path, output_json_path):
        """
        Process video file with enhanced recognition
        
        Args:
            video_path: Input video file path
            output_video_path: Output video file path
            output_json_path: Output JSON file path
        """
        if not self.is_loaded:
            print("Error: Enhanced model not loaded!")
            return False
        
        # Get reference position
        ref_pos = self.get_reference_position()
        if ref_pos is None:
            print("Error: No reference position available!")
            return False
        
        ref_x, ref_y, ref_w, ref_h = ref_pos
        
        # Load template image
        if not self.detection_data or len(self.detection_data['faces']) == 0:
            print("Error: No detection data available!")
            return False
        
        template_path = self.detection_data['faces'][0]['image_path']
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        if template_img is None:
            print(f"Error: Could not load template image {template_path}")
            return False
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Define search region (expanded around reference)
        margin = 50
        search_x = max(0, ref_x - margin)
        search_y = max(0, ref_y - margin)
        search_w = min(width - search_x, ref_w + 2 * margin)
        search_h = min(height - search_y, ref_h + 2 * margin)
        search_region = (search_x, search_y, search_w, search_h)
        
        # Process frames
        results = []
        frame_count = 0
        detection_count = 0
        
        print(f"Processing {total_frames} frames with enhanced recognition...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Template matching in search region
            match = self.template_match_in_region(gray, template_img, search_region, threshold=0.5)
            
            if match:
                detection_count += 1
                
                # Extract face region
                face_x, face_y, face_w, face_h = match['x'], match['y'], match['w'], match['h']
                face_region = gray[face_y:face_y+face_h, face_x:face_x+face_w]
                
                # Enhanced face recognition
                person_id, person_name, recognition_confidence, angle_type = self.enhanced_recognize_face(face_region)
                
                # Draw bounding box and label
                color = (0, 255, 0) if person_id != -1 else (0, 0, 255)
                cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), color, 2)
                
                # Enhanced label with angle information
                label = f"{person_name} ({recognition_confidence:.2f}) TM:{match['confidence']:.2f} [{angle_type}]"
                cv2.putText(frame, label, (face_x, face_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Store result
                results.append({
                    'frame': frame_count,
                    'person_id': int(person_id),
                    'person_name': person_name,
                    'recognition_confidence': float(recognition_confidence),
                    'template_confidence': float(match['confidence']),
                    'angle_type': angle_type,
                    'bbox': {'x': face_x, 'y': face_y, 'w': face_w, 'h': face_h}
                })
            
            # Write frame
            out.write(frame)
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Save results
        output_data = {
            'video_path': video_path,
            'total_frames': frame_count,
            'detections': detection_count,
            'recognition_results': results,
            'processing_date': datetime.now().isoformat(),
            'model_type': 'enhanced',
            'search_region': search_region
        }
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nEnhanced processing completed!")
        print(f"Total frames processed: {frame_count}")
        print(f"Total detections: {detection_count}")
        print(f"Output video: {output_video_path}")
        print(f"Output JSON: {output_json_path}")
        
        return True
    
    def process_live_camera(self):
        """
        Process live camera feed with enhanced recognition
        """
        if not self.is_loaded:
            print("Error: Enhanced model not loaded!")
            return False
        
        # Get reference position
        ref_pos = self.get_reference_position()
        if ref_pos is None:
            print("Error: No reference position available!")
            return False
        
        ref_x, ref_y, ref_w, ref_h = ref_pos
        
        # Load template image
        if not self.detection_data or len(self.detection_data['faces']) == 0:
            print("Error: No detection data available!")
            return False
        
        template_path = self.detection_data['faces'][0]['image_path']
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        if template_img is None:
            print(f"Error: Could not load template image {template_path}")
            return False
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define search region
        margin = 50
        search_x = max(0, ref_x - margin)
        search_y = max(0, ref_y - margin)
        search_w = min(width - search_x, ref_w + 2 * margin)
        search_h = min(height - search_y, ref_h + 2 * margin)
        search_region = (search_x, search_y, search_w, search_h)
        
        print("Enhanced live recognition started. Press 'q' to quit.")
        print("Features: Multi-scale, HOG, LBP, Angle detection, Profile optimization")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Template matching in search region
            match = self.template_match_in_region(gray, template_img, search_region, threshold=0.5)
            
            if match:
                # Extract face region
                face_x, face_y, face_w, face_h = match['x'], match['y'], match['w'], match['h']
                face_region = gray[face_y:face_y+face_h, face_x:face_x+face_w]
                
                # Enhanced face recognition
                person_id, person_name, recognition_confidence, angle_type = self.enhanced_recognize_face(face_region)
                
                # Draw bounding box and label
                color = (0, 255, 0) if person_id != -1 else (0, 0, 255)
                cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), color, 2)
                
                # Enhanced label with angle information
                label = f"{person_name} ({recognition_confidence:.2f}) [{angle_type}]"
                cv2.putText(frame, label, (face_x, face_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Additional info
                info_text = f"TM: {match['confidence']:.2f}"
                cv2.putText(frame, info_text, (face_x, face_y + face_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw search region
            cv2.rectangle(frame, (search_x, search_y), (search_x + search_w, search_y + search_h), (255, 255, 0), 1)
            
            # Show frame
            cv2.imshow('Enhanced Face Recognition', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("Enhanced live recognition stopped.")
        return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced face recognition scanner')
    parser.add_argument('--video', help='Input video file path')
    parser.add_argument('--live', action='store_true', help='Use live camera feed')
    parser.add_argument('--person', required=True, help='Person name for loading model and data')
    args = parser.parse_args()
    
    # Configure paths based on person name
    person_name = args.person
    model_path = f"faces/lock_version/{person_name}/face_model_enhanced.pkl"
    json_path = f"faces/lock_version/{person_name}/{person_name}_faces_detection.json"
    
    # Check if enhanced model exists
    if not os.path.exists(model_path):
        print(f"Error: Enhanced model {model_path} not found!")
        print("Please run train-enhanced.py first to create the enhanced model.")
        return
    
    # Create enhanced scanner
    scanner = EnhancedFaceScanner()
    
    # Load enhanced model
    if not scanner.load_enhanced_model(model_path):
        print("Failed to load enhanced model!")
        return
    
    # Load detection data
    if not scanner.load_detection_data(json_path):
        print("Failed to load detection data!")
        return
    
    # Process based on input mode
    if args.live:
        # Live camera mode
        scanner.process_live_camera()
    elif args.video:
        # Video file mode
        output_video_path = f"faces/lock_version/{person_name}/recognition_output_enhanced.mp4"
        output_json_path = f"faces/lock_version/{person_name}/recognition_results_enhanced.json"
        
        scanner.process_video(args.video, output_video_path, output_json_path)
    else:
        print("Error: Please specify either --video or --live mode")
        return

if __name__ == "__main__":
    main()