import cv2
import json
import os
import numpy as np
import pickle
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import image
from skimage.feature import hog, local_binary_pattern
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedFaceTrainer:
    def __init__(self, n_components=100):
        """
        Initialize enhanced face trainer with multiple feature extraction methods
        
        Args:
            n_components: Number of features after PCA dimensionality reduction
        """
        self.n_components = n_components
        
        # Multiple PCA models for different scales and features
        self.pca_models = {
            'scale_48': PCA(n_components=min(50, n_components//2)),
            'scale_64': PCA(n_components=n_components),
            'scale_80': PCA(n_components=min(80, n_components)),
            'hog': PCA(n_components=min(100, n_components)),
            'lbp': PCA(n_components=min(50, n_components//2))
        }
        
        self.scalers = {
            'scale_48': StandardScaler(),
            'scale_64': StandardScaler(),
            'scale_80': StandardScaler(),
            'hog': StandardScaler(),
            'lbp': StandardScaler()
        }
        
        self.face_features = {}
        self.face_labels = []
        self.face_info = []
        self.is_trained = False
        self.angle_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
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
        # HOG parameters optimized for face recognition
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
        # LBP parameters
        radius = 3
        n_points = 8 * radius
        
        lbp = local_binary_pattern(face_img, n_points, radius, method='uniform')
        
        # Calculate histogram
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist
    
    def augment_face_image(self, face_img):
        """
        Apply data augmentation to face image
        
        Args:
            face_img: Original face image
            
        Returns:
            List of augmented images
        """
        augmented = [face_img]  # Original image
        
        # Horizontal flip (for profile faces)
        flipped = cv2.flip(face_img, 1)
        augmented.append(flipped)
        
        # Slight rotation (-10 to +10 degrees)
        h, w = face_img.shape
        center = (w//2, h//2)
        
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(face_img, M, (w, h))
            augmented.append(rotated)
        
        # Brightness adjustment
        bright = cv2.convertScaleAbs(face_img, alpha=1.2, beta=10)
        dark = cv2.convertScaleAbs(face_img, alpha=0.8, beta=-10)
        augmented.extend([bright, dark])
        
        # Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(face_img, (3, 3), 0)
        augmented.append(blurred)
        
        return augmented
    
    def extract_multiscale_features(self, face_img, angle_type='frontal'):
        """
        Extract features at multiple scales and using different methods
        
        Args:
            face_img: Face image in grayscale
            angle_type: Type of face angle ('frontal', 'left_profile', 'right_profile')
            
        Returns:
            Dictionary of feature vectors
        """
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
        
        return features
    
    def load_face_images(self, json_path, face_dir):
        """
        Load face data with enhanced preprocessing and augmentation
        
        Args:
            json_path: JSON file path generated by detection-v2.py
            face_dir: Face images directory
        """
        print(f"Loading face data from {json_path}")
        
        # Read JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        faces_data = data['faces']
        print(f"Found {len(faces_data)} faces in JSON")
        
        # Initialize feature containers
        for key in self.pca_models.keys():
            self.face_features[key] = []
        
        valid_faces = []
        total_processed = 0
        
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
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect face angle
            angle_type = self.detect_face_angle(gray)
            
            # Apply data augmentation
            augmented_images = self.augment_face_image(gray)
            
            for aug_idx, aug_img in enumerate(augmented_images):
                # Extract multi-scale features
                features = self.extract_multiscale_features(aug_img, angle_type)
                
                # Store features
                for feature_type, feature_vector in features.items():
                    self.face_features[feature_type].append(feature_vector)
                
                # Store face info (only for original image)
                if aug_idx == 0:
                    face_info_copy = face_info.copy()
                    face_info_copy['angle_type'] = angle_type
                    valid_faces.append(face_info_copy)
                
                total_processed += 1
        
        print(f"Successfully processed {total_processed} face images (including augmentation)")
        print(f"Original faces: {len(valid_faces)}")
        
        # Convert to numpy arrays
        for key in self.face_features.keys():
            if self.face_features[key]:
                self.face_features[key] = np.array(self.face_features[key])
            else:
                self.face_features[key] = np.array([])
        
        self.face_info = valid_faces
        
        return len(valid_faces)
    
    def assign_labels_interactive(self, person_name):
        """
        Assign labels to faces with enhanced labeling for different angles
        
        Args:
            person_name: Name of the person for labeling all faces
        """
        print("\n=== Enhanced Face Labeling ===")
        print(f"Found {len(self.face_info)} faces to label.")
        print(f"Using person name: {person_name}")
        
        # Count faces by angle type
        angle_counts = {'frontal': 0, 'left_profile': 0, 'right_profile': 0}
        for face_info in self.face_info:
            angle_type = face_info.get('angle_type', 'frontal')
            angle_counts[angle_type] += 1
        
        print(f"Face angle distribution:")
        for angle, count in angle_counts.items():
            print(f"  {angle}: {count} faces")
        
        # Assign labels (considering augmentation multiplier)
        augmentation_factor = 6  # Number of augmented versions per original
        total_samples = len(self.face_info) * augmentation_factor
        
        labels = []
        person_id_map = {person_name: 0}
        
        for i in range(total_samples):
            labels.append(0)  # All faces use ID 0
        
        # Update face_info
        for i, face_info in enumerate(self.face_info):
            self.face_info[i]['person_name'] = person_name
            self.face_info[i]['person_id'] = 0
        
        print(f"\nLabeling completed!")
        print(f"Total training samples (with augmentation): {total_samples}")
        print(f"  {person_name} (ID: 0): {total_samples} samples")
        
        self.face_labels = np.array(labels)
        self.person_id_map = person_id_map
        
        return labels
    
    def train_enhanced_models(self):
        """
        Train multiple PCA models with enhanced features
        """
        if not self.face_features or len(self.face_labels) == 0:
            print("Error: No face features or labels available!")
            return False

        print(f"\nTraining enhanced PCA models...")
        print(f"Total training samples: {len(self.face_labels)}")
        
        trained_models = {}
        
        for feature_type, features in self.face_features.items():
            if len(features) == 0:
                print(f"Warning: No features for {feature_type}, skipping...")
                continue
            
            print(f"\nTraining {feature_type} model:")
            print(f"  Feature dimension: {features.shape[1]}")
            print(f"  Samples: {features.shape[0]}")
            
            # Standardize features
            scaler = self.scalers[feature_type]
            scaled_features = scaler.fit_transform(features)
            
            # Determine optimal PCA components
            n_samples, n_features = scaled_features.shape
            max_components = min(n_samples, n_features) - 1
            optimal_components = min(self.n_components, max_components)
            
            print(f"  Max possible components: {max_components}")
            print(f"  Using components: {optimal_components}")
            
            # Update PCA with optimal components
            pca = PCA(n_components=optimal_components, random_state=42)
            self.pca_models[feature_type] = pca
            
            # PCA dimensionality reduction
            pca_features = pca.fit_transform(scaled_features)
            
            print(f"  PCA components: {pca.n_components_}")
            print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
            
            trained_models[feature_type] = {
                'features': pca_features,
                'pca': pca,
                'scaler': scaler
            }
        
        self.trained_models = trained_models
        self.is_trained = True
        
        return True
    
    def save_enhanced_model(self, model_path):
        """
        Save enhanced trained model
        
        Args:
            model_path: Model save path
        """
        if not self.is_trained:
            print("Error: Model not trained yet!")
            return False
        
        model_data = {
            'trained_models': self.trained_models,
            'pca_models': self.pca_models,
            'scalers': self.scalers,
            'face_labels': self.face_labels,
            'face_info': self.face_info,
            'person_id_map': self.person_id_map,
            'n_components': self.n_components,
            'training_date': datetime.now().isoformat(),
            'model_type': 'enhanced',
            'augmentation_factor': 6
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Enhanced model saved to {model_path}")
        return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train enhanced face recognition model')
    parser.add_argument('--person', required=True, help='Person name for training model')
    args = parser.parse_args()
    
    # Configure paths based on person name
    person_name = args.person
    json_path = f"faces/lock_version/{person_name}/{person_name}_faces_detection.json"
    face_dir = f"faces/lock_version/{person_name}"
    model_path = f"faces/lock_version/{person_name}/face_model_enhanced.pkl"
    
    # Check input file
    if not os.path.exists(json_path):
        print(f"Error: JSON file {json_path} not found!")
        print("Please run detection-v2.py first to generate face data.")
        return
    
    # Create enhanced trainer
    trainer = EnhancedFaceTrainer(n_components=100)
    
    # Load face images with enhancement
    num_faces = trainer.load_face_images(json_path, face_dir)
    if num_faces == 0:
        print("No valid face images found!")
        return
    
    # Assign labels to faces
    labels = trainer.assign_labels_interactive(person_name)
    if len(labels) == 0:
        print("No faces labeled, training cancelled.")
        return
    
    # Train enhanced models
    if trainer.train_enhanced_models():
        # Save enhanced model
        trainer.save_enhanced_model(model_path)
        print(f"\nEnhanced training completed successfully!")
        print(f"Enhanced model saved to: {model_path}")
        print(f"\nModel features:")
        print(f"  - Multi-scale feature extraction (48x48, 64x64, 80x80)")
        print(f"  - HOG and LBP features")
        print(f"  - Data augmentation (6x samples)")
        print(f"  - Angle-aware preprocessing")
        print(f"  - Enhanced profile face handling")
    else:
        print("Enhanced training failed!")

if __name__ == "__main__":
    main()