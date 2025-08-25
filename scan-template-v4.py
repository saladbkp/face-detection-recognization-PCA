import cv2
import json
import os
import numpy as np
import pickle
import glob
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class MultiModelFaceScanner:
    def __init__(self):
        """
        Initialize multi-model face scanner
        """
        self.models = {}
        
    def load_all_models(self):
        """
        Load all available models from faces/lock_version/*/face_model.pkl
        """
        model_pattern = "faces/lock_version/*/face_model.pkl"
        model_paths = glob.glob(model_pattern)
        
        if not model_paths:
            print(f"No models found matching pattern: {model_pattern}")
            return False
        
        print(f"Found {len(model_paths)} model(s):")
        
        for model_path in model_paths:
            # Extract person name from path
            person_name = os.path.basename(os.path.dirname(model_path))
            
            try:
                # Load model
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Load detection data
                detection_json_path = os.path.join(os.path.dirname(model_path), f"{person_name}_faces_detection.json")
                detection_data = None
                if os.path.exists(detection_json_path):
                    with open(detection_json_path, 'r', encoding='utf-8') as f:
                        detection_data = json.load(f)
                
                # Load template images (first 5 faces for better matching)
                template_images = []
                if detection_data and detection_data['faces']:
                    for i, face_data in enumerate(detection_data['faces'][:5]):
                        template_path = face_data['image_path']
                        if os.path.exists(template_path):
                            template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                            if template_img is not None:
                                template_images.append({
                                    'image': template_img,
                                    'width': face_data['width'],
                                    'height': face_data['height']
                                })
                
                self.models[person_name] = {
                    'model_data': model_data,
                    'detection_data': detection_data,
                    'template_images': template_images,
                    'model_path': model_path
                }
                
                face_count = len(model_data['face_features']) if model_data else 0
                print(f"  - {person_name}: {face_count} faces")
                
            except Exception as e:
                print(f"  - Failed to load {person_name}: {e}")
        
        print(f"Successfully loaded {len(self.models)} model(s)")
        return len(self.models) > 0
    
    def is_detection_in_corner(self, detection, frame_width, frame_height, corner_threshold=0.15, border_threshold=0.05):
        """
        Check if detection is in corner area or border area of the frame
        
        Args:
            detection: Detection dictionary with x, y, width, height
            frame_width: Frame width
            frame_height: Frame height
            corner_threshold: Threshold ratio for corner area (0.15 = 15% of frame)
            border_threshold: Threshold ratio for border area (0.05 = 5% of frame)
            
        Returns:
            bool: True if detection is in corner area or border area
        """
        x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
        
        # Calculate corner boundaries
        corner_w = int(frame_width * corner_threshold)
        corner_h = int(frame_height * corner_threshold)
        
        # Calculate border boundaries
        border_w = int(frame_width * border_threshold)
        border_h = int(frame_height * border_threshold)
        
        # Check if detection center is in any corner
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Check if detection touches any border
        if (x < border_w or  # Left border
            y < border_h or  # Top border
            (x + w) > (frame_width - border_w) or  # Right border
            (y + h) > (frame_height - border_h)):  # Bottom border
            return True
        
        # Top-left corner
        if center_x < corner_w and center_y < corner_h:
            return True
        
        # Top-right corner
        if center_x > (frame_width - corner_w) and center_y < corner_h:
            return True
        
        # Bottom-left corner
        if center_x < corner_w and center_y > (frame_height - corner_h):
            return True
        
        # Bottom-right corner
        if center_x > (frame_width - corner_w) and center_y > (frame_height - corner_h):
            return True
        
        return False
    
    def template_match_all_models(self, frame):
        """
        Perform template matching using all loaded models
        
        Args:
            frame: Current frame (grayscale)
            
        Returns:
            list: List of detected faces with person names and confidence scores
        """
        detected_faces = []
        frame_height, frame_width = frame.shape[:2]
        
        for person_name, model_info in self.models.items():
            template_images = model_info['template_images']
            detection_data = model_info['detection_data']
            
            if not template_images or not detection_data:
                continue
                
            # Get average face size from detection data for search area estimation
            avg_width = np.mean([face['width'] for face in detection_data['faces']])
            avg_height = np.mean([face['height'] for face in detection_data['faces']])
            
            best_match = None
            best_score = 0.0
            
            # Try each template image
            for template_info in template_images:
                template = template_info['image']
                
                # Multi-scale template matching
                for scale in [0.8, 1.0, 1.2]:
                    # Resize template
                    new_width = int(template.shape[1] * scale)
                    new_height = int(template.shape[0] * scale)
                    
                    if new_width < 20 or new_height < 20 or new_width > frame.shape[1] or new_height > frame.shape[0]:
                        continue
                        
                    scaled_template = cv2.resize(template, (new_width, new_height))
                    
                    # Template matching
                    result = cv2.matchTemplate(frame, scaled_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_score:
                        candidate_match = {
                            'x': max_loc[0],
                            'y': max_loc[1],
                            'width': new_width,
                            'height': new_height,
                            'person_name': person_name,
                            'confidence': max_val,
                            'scale': scale
                        }
                        
                        # Check if detection is in corner - if so, skip it
                        if not self.is_detection_in_corner(candidate_match, frame_width, frame_height):
                            best_score = max_val
                            best_match = candidate_match
            
            # Add best match if confidence is above threshold and not in corner
            if best_match and best_score > 0.6:
                detected_faces.append(best_match)
        
        # Return all detected faces for further processing
        # Let process_live_camera method handle the selection of best face
        return detected_faces
    
    def non_max_suppression(self, detections, overlap_threshold=0.3):
        """
        Apply non-maximum suppression to remove overlapping detections
        """
        if len(detections) == 0:
            return []
            
        # Sort by confidence score
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        while detections:
            # Take the detection with highest confidence
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            remaining = []
            for det in detections:
                if self.calculate_overlap(current, det) < overlap_threshold:
                    remaining.append(det)
            detections = remaining
            
        return keep
    
    def calculate_overlap(self, det1, det2):
        """
        Calculate overlap ratio between two detections
        """
        x1_min, y1_min = det1['x'], det1['y']
        x1_max, y1_max = x1_min + det1['width'], y1_min + det1['height']
        
        x2_min, y2_min = det2['x'], det2['y']
        x2_max, y2_max = x2_min + det2['width'], y2_min + det2['height']
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
            
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = det1['width'] * det1['height']
        area2 = det2['width'] * det2['height']
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def extract_face_features(self, face_img, model_data):
        """
        Extract face features using specific model
        """
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        resized = cv2.resize(gray, (64, 64))
        flattened = resized.flatten().reshape(1, -1)
        
        scaled = model_data['scaler'].transform(flattened)
        features = model_data['pca'].transform(scaled)
        
        return features[0]
    
    def recognize_face_with_model(self, face_features, model_data, threshold=0.7):
        """
        Recognize face using specific model
        """
        similarities = cosine_similarity([face_features], model_data['face_features'])[0]
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        
        if max_similarity >= threshold:
            person_id = model_data['face_labels'][max_idx]
            person_name = "unknown"
            for name, pid in model_data['person_id_map'].items():
                if pid == person_id:
                    person_name = name
                    break
            return person_id, person_name, max_similarity
        else:
            return -1, "unknown", max_similarity
    
    def recognize_face_all_models(self, face_img, threshold=0.8):
        """
        Recognize face using all loaded models and return best match
        """
        best_result = None
        best_confidence = 0.0
        best_person = "unknown"
        
        for person_name, model_info in self.models.items():
            model_data = model_info['model_data']
            if model_data is None:
                continue
            
            try:
                features = self.extract_face_features(face_img, model_data)
                person_id, recognized_name, confidence = self.recognize_face_with_model(features, model_data, threshold)
                print(person_id,recognized_name,confidence,threshold)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_person = recognized_name if recognized_name != "unknown" else person_name
                    best_result = (person_id, best_person, confidence)
                    print("this is the best",best_result)
            
            except Exception as e:
                print(f"Error recognizing with model {person_name}: {e}")
                continue
        
        if best_result:
            return best_result
        else:
            return -1, "unknown", 0.0
    
    def process_live_camera(self, show_preview=True):
        """
        Process live camera for real-time face recognition using template matching
        """
        if not self.models:
            print("Error: No models loaded!")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting live face recognition with template matching...")
        print("Press 'q' to quit")
        
        frame_count = 0
        recognition_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using template matching
            detected_faces = self.template_match_all_models(gray)
            
            # If multiple faces detected, select the one with largest size and highest PCA confidence
            if len(detected_faces) > 1:
                best_detection = None
                best_score = -1
                
                for detection in detected_faces:
                    x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
                    
                    # Extract face image for PCA verification
                    face_img = frame[y:y+h, x:x+w]
                    person_id, pca_person_name, pca_confidence = self.recognize_face_all_models(face_img)
                    
                    # Calculate size (area) of the detection
                    size = w * h
                    
                    # Combined score: prioritize size and PCA confidence
                    # Normalize size to 0-1 range (assuming max reasonable face size is 200x200)
                    normalized_size = min(size / (200 * 200), 1.0)
                    combined_score = normalized_size * 0.5 + pca_confidence * 0.5
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_detection = detection
                        best_detection['pca_confidence'] = pca_confidence
                        best_detection['pca_person_name'] = pca_person_name
                
                detected_faces = [best_detection] if best_detection else []
            
            for detection in detected_faces:
                x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
                person_name = detection['person_name']
                template_confidence = detection['confidence']
                
                # Get PCA results (either from selection above or compute now)
                if 'pca_confidence' in detection:
                    pca_confidence = detection['pca_confidence']
                    pca_person_name = detection['pca_person_name']
                else:
                    # Extract face image for PCA verification
                    face_img = frame[y:y+h, x:x+w]
                    person_id, pca_person_name, pca_confidence = self.recognize_face_all_models(face_img)
                
                # Use template matching result if PCA verification matches or has low confidence
                if pca_person_name == person_name or pca_confidence < 0.5:
                    final_person_name = person_name
                    final_confidence = template_confidence
                else:
                    final_person_name = pca_person_name
                    final_confidence = pca_confidence
                if pca_confidence < 0.8 or template_confidence<0.7:
                    final_person_name = "unknown"
                # print(">>>>",detection)
                print(person_name,template_confidence,pca_person_name,pca_confidence)
                # Draw result
                color = (0, 255, 0) if final_person_name != "unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Add label with both template and PCA confidence
                label = f"{final_person_name} (T:{template_confidence:.2f}, P:{pca_confidence:.2f})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Record result
                result = {
                    'frame_number': frame_count,
                    'person_name': final_person_name,
                    'template_confidence': template_confidence,
                    'pca_confidence': pca_confidence,
                    'final_confidence': final_confidence,
                    'x': x, 'y': y, 'width': w, 'height': h
                }
                print(result)
                recognition_results.append(result)
            
            # Show preview
            if show_preview:
                cv2.imshow('Template Matching Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        
        print("\nLive recognition stopped!")
        return recognition_results


def main():
    # Create multi-model scanner
    scanner = MultiModelFaceScanner()
    
    # Load all available models
    if not scanner.load_all_models():
        print("Failed to load any models!")
        print("Please run train-v4.py first to train models.")
        return
    
    # Start live camera recognition
    print("\nStarting live face recognition...")
    print("Press 'q' to quit")
    results = scanner.process_live_camera(show_preview=True)
        
    if results:
        print(f"\nRecognition summary:")
        # Count recognition results
        person_counts = {}
        for result in results:
            name = result['person_name']
            person_counts[name] = person_counts.get(name, 0) + 1
        
        for name, count in person_counts.items():
            print(f"  {name}: {count} detections")

if __name__ == "__main__":
    main()