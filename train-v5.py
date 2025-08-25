import cv2
import json
import os
import numpy as np
import pickle
import glob
import re
import subprocess
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class MultiFaceTrainer:
    def __init__(self, n_components=50):
        """
        Initialize multi-person face trainer with eigenfaces support
        
        Args:
            n_components: Number of features after PCA dimensionality reduction
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.face_features = []
        self.face_labels = []
        self.face_info = []
        self.is_trained = False
        self.mean_face = None
        self.eigenfaces = None
        self.face_shape = (64, 64)  # Standard face image size
        self.person_id_map = {}
        
    def generate_detection_json_for_person(self, person_name, face_dir):
        """
        Generate detection JSON file for a person based on existing face images
        
        Args:
            person_name: Name of the person
            face_dir: Directory containing face images
        """
        print(f"Generating detection JSON for {person_name}...")
        
        # Find all face images in the directory
        face_images = []
        
        # Look for different naming patterns
        patterns = [
            f"{person_name}_face_*.jpg",
            "face_*_frame_*.jpg",
            "*.jpg"
        ]
        
        for pattern in patterns:
            search_path = os.path.join(face_dir, pattern)
            found_files = glob.glob(search_path)
            for file_path in found_files:
                filename = os.path.basename(file_path)
                # Skip non-face images (like eigenfaces, mean_face, etc.)
                if any(skip in filename.lower() for skip in ['eigenface', 'mean_face', 'model_info']):
                    continue
                face_images.append(file_path)
        
        # Remove duplicates and sort
        face_images = sorted(list(set(face_images)))
        
        print(f"Found {len(face_images)} face images for {person_name}")
        
        if len(face_images) == 0:
            print(f"No face images found for {person_name}")
            return None
        
        # Generate face data
        all_face_data = []
        
        for face_id, image_path in enumerate(face_images):
            filename = os.path.basename(image_path)
            
            # Try to extract frame number from filename
            frame_number = 0
            
            # Pattern 1: face_XXXXXX_frame_XXXXXX.jpg
            match = re.search(r'face_\d+_frame_(\d+)', filename)
            if match:
                frame_number = int(match.group(1))
            
            # Pattern 2: person_face_XXXX.jpg
            elif re.search(r'_face_(\d+)', filename):
                match = re.search(r'_face_(\d+)', filename)
                frame_number = int(match.group(1))
            
            # Try to get image dimensions
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    height, width = img.shape[:2]
                else:
                    width, height = 64, 64  # Default size
            except:
                width, height = 64, 64  # Default size
            
            # Calculate timestamp (assume 30 FPS)
            fps = 30.0
            timestamp = frame_number / fps
            
            # Create face data entry
            face_data = {
                "face_id": face_id,
                "frame_number": frame_number,
                "timestamp": timestamp,
                "x": 0,  # Unknown, set to 0
                "y": 0,  # Unknown, set to 0
                "width": width,
                "height": height,
                "center_x": width // 2,
                "center_y": height // 2,
                "area": width * height,
                "image_path": image_path,
                "image_filename": filename
            }
            
            all_face_data.append(face_data)
        
        # Create video info structure
        video_info = {
            "video_path": f"videos/{person_name}.mp4",  # Assumed video path
            "total_frames": max([face['frame_number'] for face in all_face_data]) + 1 if all_face_data else 0,
            "fps": fps,
            "total_faces_detected": len(all_face_data),
            "processing_date": datetime.now().isoformat(),
            "faces": all_face_data
        }
        
        # Save JSON file
        json_path = os.path.join(face_dir, f"{person_name}_faces_detection.json")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(video_info, f, indent=2, ensure_ascii=False)
        
        print(f"Generated {json_path}")
        print(f"Total faces: {len(all_face_data)}")
        
        return json_path
    
    def count_face_images_in_directory(self, directory):
        """
        Count total number of face images in a directory
        
        Args:
            directory: Directory containing face images (can be base dir or person dir)
            
        Returns:
            Total number of face images
        """
        if not os.path.exists(directory):
            return 0
        
        # Check if this directory contains subdirectories (base directory case)
        subdirs = [d for d in os.listdir(directory) 
                   if os.path.isdir(os.path.join(directory, d))]
        
        # If there are subdirectories, this is a base directory
        if subdirs and any(os.path.exists(os.path.join(directory, d, f"{d}_faces_detection.json")) or 
                          glob.glob(os.path.join(directory, d, "*.jpg")) for d in subdirs):
            total_images = 0
            for person_name in subdirs:
                person_dir = os.path.join(directory, person_name)
                person_count = self._count_face_images_in_single_dir(person_dir)
                total_images += person_count
                print(f"{person_name}: {person_count} face images")
            return total_images
        else:
            # This is a single person directory
            return self._count_face_images_in_single_dir(directory)
    
    def _count_face_images_in_single_dir(self, person_dir):
        """
        Count face images in a single person directory
        
        Args:
            person_dir: Person's directory
        """
        if not os.path.exists(person_dir):
            return 0
        
        # Count JPG files, excluding eigenfaces and mean_face
        jpg_files = glob.glob(os.path.join(person_dir, "*.jpg"))
        face_images = [f for f in jpg_files 
                      if not any(skip in os.path.basename(f).lower() 
                               for skip in ['eigenface', 'mean_face', 'model_info'])]
        
        return len(face_images)
    
    def load_all_face_images(self, base_dir):
        """
        Load face data from all person directories
        
        Args:
            base_dir: Base directory containing person folders
        """
        print(f"Loading face data from all persons in {base_dir}")
        
        if not os.path.exists(base_dir):
            print(f"Error: Directory {base_dir} not found!")
            return 0
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d))]
        
        print(f"Found {len(person_dirs)} person directories: {person_dirs}")
        
        all_face_images = []
        all_face_info = []
        person_id = 0
        
        for person_name in person_dirs:
            person_dir = os.path.join(base_dir, person_name)
            json_path = os.path.join(person_dir, f"{person_name}_faces_detection.json")
            
            # Generate JSON file if it doesn't exist
            if not os.path.exists(json_path):
                print(f"JSON file not found for {person_name}, generating...")
                self.generate_detection_json_for_person(person_name, person_dir)
            
            # Load JSON data
            if not os.path.exists(json_path):
                print(f"Warning: Could not generate JSON for {person_name}, skipping...")
                continue
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            faces_data = data['faces']
            print(f"Found {len(faces_data)} faces for {person_name}")
            
            # Add person to ID mapping
            self.person_id_map[person_name] = person_id
            
            for i, face_info in enumerate(faces_data):
                image_path = face_info['image_path']
                
                # Check if image file exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} not found, skipping...")
                    continue
                
                # Read and preprocess image
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Could not read image {image_path}, skipping...")
                    continue
                
                # Convert to grayscale and resize
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, self.face_shape)  # Uniform size
                
                all_face_images.append(resized.flatten())  # Flatten to 1D vector
                
                # Update face info with person information
                face_info['person_name'] = person_name
                face_info['person_id'] = person_id
                all_face_info.append(face_info)
            
            person_id += 1
        
        print(f"Successfully loaded {len(all_face_images)} face images from {len(self.person_id_map)} persons")
        
        self.face_images = np.array(all_face_images)
        self.face_info = all_face_info
        
        # Create labels array
        self.face_labels = np.array([info['person_id'] for info in all_face_info])
        
        return len(all_face_images)
    
    def load_face_images_from_json(self, json_path, person_dir):
        """
        Load face images for a single person from JSON file
        
        Args:
            json_path: JSON file path
            person_dir: Person's directory containing face images
        """
        print(f"Loading face data from {json_path}")
        
        if not os.path.exists(json_path):
            print(f"Error: JSON file {json_path} not found!")
            return 0
        
        # Read JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        faces_data = data['faces']
        print(f"Found {len(faces_data)} faces in JSON")
        
        face_images = []
        valid_faces = []
        
        for i, face_info in enumerate(faces_data):
            # Try different ways to construct image path
            image_path = None
            
            # Method 1: Use image_filename if available
            if 'image_filename' in face_info:
                image_path = os.path.join(person_dir, face_info['image_filename'])
            # Method 2: Use image_path directly
            elif 'image_path' in face_info:
                image_path = face_info['image_path']
            # Method 3: Fallback - construct from face_id and frame_number
            else:
                face_id = face_info.get('face_id', i)
                frame_number = face_info.get('frame_number', i)
                filename = f"face_{face_id}_frame_{frame_number}.jpg"
                image_path = os.path.join(person_dir, filename)
            
            # Check if image file exists
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, skipping...")
                continue
            
            # Read and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}, skipping...")
                continue
            
            # Convert to grayscale and resize
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, self.face_shape)  # Uniform size
            
            face_images.append(resized.flatten())  # Flatten to 1D vector
            valid_faces.append(face_info)
        
        print(f"Successfully loaded {len(face_images)} face images")
        
        self.face_images = np.array(face_images)
        self.face_info = valid_faces
        
        # Create labels array (all faces belong to the same person, so label = 0)
        self.face_labels = np.zeros(len(face_images), dtype=int)
        
        # Set person ID mapping for single person
        person_name = os.path.basename(person_dir)
        self.person_id_map = {person_name: 0}
        
        return len(face_images)
    
    def train_pca_model(self):
        """
        Train face feature model using PCA and generate eigenfaces
        """
        if len(self.face_images) == 0:
            print("Error: No face images loaded!")
            return False
        
        if len(self.face_labels) == 0:
            print("Error: No face labels assigned!")
            return False
        
        print(f"\nTraining PCA model with {len(self.face_images)} faces...")
        print(f"Original feature dimension: {self.face_images.shape[1]}")
        print(f"Reducing to {self.n_components} components")
        
        # Calculate mean face
        self.mean_face = np.mean(self.face_images, axis=0)
        print(f"Mean face calculated with shape: {self.mean_face.shape}")
        
        # Standardize features
        scaled_features = self.scaler.fit_transform(self.face_images)
        
        # PCA dimensionality reduction
        pca_features = self.pca.fit_transform(scaled_features)
        
        # Extract eigenfaces (principal components)
        self.eigenfaces = self.pca.components_
        print(f"Generated {len(self.eigenfaces)} eigenfaces")
        
        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        print(f"Reduced feature dimension: {pca_features.shape[1]}")
        
        self.face_features = pca_features
        self.is_trained = True
        
        return True
    
    def save_eigenfaces(self, output_dir):
        """
        Save eigenfaces and mean face as images
        
        Args:
            output_dir: Output directory for saving images
        """
        if not self.is_trained:
            print("Error: Model not trained yet!")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save mean face
        mean_face_img = self.mean_face.reshape(self.face_shape)
        mean_face_normalized = cv2.normalize(mean_face_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mean_face_path = os.path.join(output_dir, "multi_person_mean_face.jpg")
        cv2.imwrite(mean_face_path, mean_face_normalized)
        print(f"Mean face saved to: {mean_face_path}")
        
        # Save top eigenfaces
        num_eigenfaces_to_save = min(10, len(self.eigenfaces))
        for i in range(num_eigenfaces_to_save):
            eigenface = self.eigenfaces[i].reshape(self.face_shape)
            # Normalize eigenface for visualization
            eigenface_normalized = cv2.normalize(eigenface, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            eigenface_path = os.path.join(output_dir, f"multi_person_eigenface_{i+1:02d}.jpg")
            cv2.imwrite(eigenface_path, eigenface_normalized)
        
        print(f"Saved {num_eigenfaces_to_save} eigenfaces to {output_dir}")
        
        # Save model information
        model_info = {
            'training_date': datetime.now().isoformat(),
            'total_faces': len(self.face_images),
            'total_persons': len(self.person_id_map),
            'person_id_map': self.person_id_map,
            'n_components': self.n_components,
            'explained_variance_ratio': float(self.pca.explained_variance_ratio_.sum()),
            'face_shape': self.face_shape,
            'eigenfaces_saved': num_eigenfaces_to_save
        }
        
        info_path = os.path.join(output_dir, "multi_person_model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"Model information saved to: {info_path}")
        return True
    
    def save_model(self, model_path):
        """
        Save trained model with eigenfaces support
        
        Args:
            model_path: Model save path
        """
        if not self.is_trained:
            print("Error: Model not trained yet!")
            return False
        
        model_data = {
            'pca': self.pca,
            'scaler': self.scaler,
            'face_features': self.face_features,
            'face_labels': self.face_labels,
            'face_info': self.face_info,
            'person_id_map': self.person_id_map,
            'n_components': self.n_components,
            'mean_face': self.mean_face,
            'eigenfaces': self.eigenfaces,
            'face_shape': self.face_shape,
            'training_date': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
        return True
    
    def load_model(self, model_path):
        """
        Load trained model with eigenfaces support
        
        Args:
            model_path: Model file path
        """
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found!")
            return False
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.face_features = model_data['face_features']
        self.face_labels = model_data['face_labels']
        self.face_info = model_data['face_info']
        self.person_id_map = model_data['person_id_map']
        self.n_components = model_data['n_components']
        
        # Load eigenfaces data if available
        self.mean_face = model_data.get('mean_face', None)
        self.eigenfaces = model_data.get('eigenfaces', None)
        self.face_shape = model_data.get('face_shape', (64, 64))
        
        self.is_trained = True
        
        print(f"Model loaded from {model_path}")
        print(f"Training date: {model_data.get('training_date', 'Unknown')}")
        print(f"Total faces: {len(self.face_features)}")
        print(f"Total persons: {len(self.person_id_map)}")
        if self.eigenfaces is not None:
            print(f"Eigenfaces loaded: {len(self.eigenfaces)}")
        
        return True

def train_person_model(person_name, base_dir):
    """
    Train model for a specific person
    
    Args:
        person_name: Name of the person to train
        base_dir: Base directory containing person folders
    """
    print(f"\n=== Training model for {person_name} ===")
    
    # Configure paths for this person
    person_dir = os.path.join(base_dir, person_name)
    json_file = os.path.join(person_dir, f"{person_name}_faces_detection.json")
    model_path = os.path.join(person_dir, "face_model.pkl")
    
    # Check if person directory exists
    if not os.path.exists(person_dir):
        print(f"Error: Person directory {person_dir} not found!")
        return False
    
    # Count face images for this person
    trainer_temp = MultiFaceTrainer()
    face_count = trainer_temp.count_face_images_in_directory(person_dir)
    
    if face_count == 0:
        print(f"No face images found for {person_name}!")
        return False
    
    # Generate JSON file if it doesn't exist
    if not os.path.exists(json_file):
        trainer_temp.generate_detection_json_for_person(person_name, person_dir)
    
    # Set n_components based on face count (use face_count for this person)
    optimal_components = face_count if face_count > 1 else 1
    print(f"Face images found for {person_name}: {face_count}")
    print(f"Setting n_components to: {optimal_components}")
    
    # Create trainer with optimal n_components
    trainer = MultiFaceTrainer(n_components=optimal_components)
    
    # Load face images for this person only
    num_faces = trainer.load_face_images_from_json(json_file, person_dir)
    if num_faces == 0:
        print(f"No valid face images loaded for {person_name}!")
        return False
    
    print(f"Loaded {num_faces} face images for {person_name}")
    
    # Train PCA model
    if trainer.train_pca_model():
        # Save eigenfaces as images in person's directory
        trainer.save_eigenfaces(person_dir)
        
        # Save model in person's directory
        trainer.save_model(model_path)
        print(f"Training completed successfully for {person_name}!")
        print(f"Model saved to: {model_path}")
        print(f"Eigenfaces and mean face saved to: {person_dir}")
        return True
    else:
        print(f"Training failed for {person_name}!")
        return False

def main():
    # Configure paths
    base_dir = "faces/lock_version"
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} not found!")
        return
    
    # Find all person directories
    person_dirs = [d for d in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, d))]
    
    if not person_dirs:
        print(f"No person directories found in {base_dir}!")
        return
    
    print(f"Found {len(person_dirs)} person directories: {person_dirs}")
    
    # Train model for each person
    successful_trainings = 0
    failed_trainings = 0
    
    for person_name in person_dirs:
        try:
            if train_person_model(person_name, base_dir):
                successful_trainings += 1
            else:
                failed_trainings += 1
        except Exception as e:
            print(f"Error training model for {person_name}: {str(e)}")
            failed_trainings += 1
    
    print(f"\n=== Training Summary ===")
    print(f"Total persons processed: {len(person_dirs)}")
    print(f"Successful trainings: {successful_trainings}")
    print(f"Failed trainings: {failed_trainings}")
    
    if successful_trainings > 0:
        print(f"\nModels saved in individual person directories under {base_dir}")
        print(f"Each model can be used with scan-template scripts for person-specific recognition")

if __name__ == "__main__":
    main()