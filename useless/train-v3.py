import cv2
import os
import numpy as np
import pickle
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import glob

class FaceTrainer:
    def __init__(self, n_components=50):
        """
        Initialize face trainer with eigenfaces support
        
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
        
    def load_face_images_from_directory(self, face_dir, person_name):
        """
        Load all face images directly from directory
        
        Args:
            face_dir: Face images directory
            person_name: Name of the person for labeling
        """
        print(f"Loading face images from {face_dir}")
        
        # Find all jpg images in the directory
        image_patterns = [
            os.path.join(face_dir, "*.jpg"),
            os.path.join(face_dir, "*.jpeg"),
            os.path.join(face_dir, "*.png")
        ]
        
        image_files = []
        for pattern in image_patterns:
            image_files.extend(glob.glob(pattern))
        
        print(f"Found {len(image_files)} image files")
        
        face_images = []
        valid_faces = []
        
        for i, image_path in enumerate(image_files):
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
            
            # Create face info
            face_info = {
                'image_path': image_path,
                'person_name': person_name,
                'person_id': 0,  # Single person, so ID is always 0
                'face_id': i
            }
            valid_faces.append(face_info)
        
        print(f"Successfully loaded {len(face_images)} face images")
        
        self.face_images = np.array(face_images)
        self.face_info = valid_faces
        
        # Assign labels (all faces belong to the same person)
        self.face_labels = np.zeros(len(face_images), dtype=int)
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
    
    def save_eigenfaces(self, output_dir, person_name):
        """
        Save eigenfaces and mean face as images
        
        Args:
            output_dir: Output directory for saving images
            person_name: Person name for file naming
        """
        if not self.is_trained:
            print("Error: Model not trained yet!")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save mean face
        mean_face_img = self.mean_face.reshape(self.face_shape)
        mean_face_normalized = cv2.normalize(mean_face_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mean_face_path = os.path.join(output_dir, f"{person_name}_mean_face.jpg")
        cv2.imwrite(mean_face_path, mean_face_normalized)
        print(f"Mean face saved to: {mean_face_path}")
        
        # Save top eigenfaces
        num_eigenfaces_to_save = min(10, len(self.eigenfaces))
        for i in range(num_eigenfaces_to_save):
            eigenface = self.eigenfaces[i].reshape(self.face_shape)
            # Normalize eigenface for visualization
            eigenface_normalized = cv2.normalize(eigenface, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            eigenface_path = os.path.join(output_dir, f"{person_name}_eigenface_{i+1:02d}.jpg")
            cv2.imwrite(eigenface_path, eigenface_normalized)
        
        print(f"Saved {num_eigenfaces_to_save} eigenfaces to {output_dir}")
        
        # Save model information
        model_info = {
            'person_name': person_name,
            'training_date': datetime.now().isoformat(),
            'total_faces': len(self.face_images),
            'n_components': self.n_components,
            'explained_variance_ratio': float(self.pca.explained_variance_ratio_.sum()),
            'face_shape': self.face_shape,
            'eigenfaces_saved': num_eigenfaces_to_save
        }
        
        info_path = os.path.join(output_dir, f"{person_name}_model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            import json
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train face recognition model using PCA from image directory')
    parser.add_argument('--person', required=True, help='Person name for training model')
    args = parser.parse_args()
    
    # Configure paths based on person name
    person_name = args.person
    face_dir = f"faces/lock_version/{person_name}"
    model_path = f"faces/lock_version/{person_name}/face_model.pkl"
    
    # Check input directory
    if not os.path.exists(face_dir):
        print(f"Error: Face directory {face_dir} not found!")
        print(f"Please create the directory and add face images for {person_name}.")
        return
    
    # Create trainer
    trainer = FaceTrainer(n_components=50)
    
    # Load face images directly from directory
    num_faces = trainer.load_face_images_from_directory(face_dir, person_name)
    if num_faces == 0:
        print("No valid face images found!")
        return
    
    print(f"\nLoaded {num_faces} face images for {person_name}")
    
    # Train PCA model
    if trainer.train_pca_model():
        # Save eigenfaces as images
        trainer.save_eigenfaces(face_dir, person_name)
        
        # Save model
        trainer.save_model(model_path)
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Eigenfaces and mean face saved to: {face_dir}")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()