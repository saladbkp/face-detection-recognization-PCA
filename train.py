import cv2
import os
import numpy as np
import pickle
import json
from datetime import datetime

def load_face_images(faces_dir):
    """
    Load all face images from the faces directory
    
    Args:
        faces_dir (str): Directory containing face images
    
    Returns:
        tuple: (face_vectors, filenames) where face_vectors is 2D array
    """
    face_vectors = []
    filenames = []
    
    print(f"Loading face images from: {faces_dir}")
    
    # Get all jpg files in the directory
    image_files = [f for f in os.listdir(faces_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        raise ValueError(f"No image files found in {faces_dir}")
    
    for filename in sorted(image_files):
        file_path = os.path.join(faces_dir, filename)
        
        # Load image in grayscale
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not load image {filename}")
            continue
        
        # Flatten image to 1D vector
        face_vector = img.flatten().astype(np.float64)
        
        face_vectors.append(face_vector)
        filenames.append(filename)
    
    if not face_vectors:
        raise ValueError("No valid face images could be loaded")
    
    # Convert to numpy array
    face_matrix = np.array(face_vectors)
    
    print(f"Loaded {len(face_vectors)} face images")
    print(f"Face vector dimension: {face_matrix.shape[1]}")
    
    return face_matrix, filenames

def manual_pca(data_matrix, n_components=None):
    """
    Manual implementation of PCA without using sklearn
    
    Args:
        data_matrix (np.array): Input data matrix (n_samples x n_features)
        n_components (int): Number of principal components to keep
    
    Returns:
        tuple: (eigenfaces, mean_face, projected_data, eigenvalues)
    """
    print("Starting manual PCA computation...")
    
    # Step 1: Calculate mean face
    mean_face = np.mean(data_matrix, axis=0)
    print(f"Mean face calculated, shape: {mean_face.shape}")
    
    # Step 2: Center the data by subtracting mean
    centered_data = data_matrix - mean_face
    print(f"Data centered, shape: {centered_data.shape}")
    
    # Step 3: Calculate covariance matrix
    # For efficiency, we use the trick: if data is (n x d) and n < d,
    # we compute covariance of (n x n) instead of (d x d)
    n_samples, n_features = centered_data.shape
    
    if n_samples < n_features:
        # Compute smaller covariance matrix (n x n)
        cov_matrix = np.dot(centered_data, centered_data.T) / (n_samples - 1)
        print(f"Using efficient covariance computation: {cov_matrix.shape}")
        
        # Step 4: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Step 5: Transform eigenvectors back to original space
        eigenfaces = np.dot(centered_data.T, eigenvectors)
        
        # Normalize eigenfaces
        for i in range(eigenfaces.shape[1]):
            eigenfaces[:, i] = eigenfaces[:, i] / np.linalg.norm(eigenfaces[:, i])
    
    else:
        # Standard covariance matrix computation
        cov_matrix = np.cov(centered_data.T)
        print(f"Standard covariance computation: {cov_matrix.shape}")
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenfaces = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenfaces = eigenfaces[:, idx]
    
    # Keep only the specified number of components
    if n_components is None:
        n_components = min(n_samples - 1, n_features)
    
    n_components = min(n_components, len(eigenvalues))
    eigenvalues = eigenvalues[:n_components]
    eigenfaces = eigenfaces[:, :n_components]
    
    print(f"Selected {n_components} principal components")
    print(f"Explained variance ratio: {eigenvalues[:5] / np.sum(eigenvalues)}")
    
    # Step 6: Project data onto eigenfaces
    projected_data = np.dot(centered_data, eigenfaces)
    
    print(f"PCA computation completed")
    print(f"Eigenfaces shape: {eigenfaces.shape}")
    print(f"Projected data shape: {projected_data.shape}")
    
    return eigenfaces, mean_face, projected_data, eigenvalues

def save_pca_model(eigenfaces, mean_face, projected_data, eigenvalues, filenames, person_name, model_dir, version=None):
    """
    Save the trained PCA model to files
    
    Args:
        eigenfaces (np.array): Principal component vectors
        mean_face (np.array): Mean face vector
        projected_data (np.array): Projected training data
        eigenvalues (np.array): Eigenvalues
        filenames (list): List of training image filenames
        person_name (str): Name of the person
        model_dir (str): Directory to save model files
        version (str): Version identifier (e.g., 'dark', 'light')
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Create model data dictionary
    model_data = {
        'eigenfaces': eigenfaces,
        'mean_face': mean_face,
        'projected_data': projected_data,
        'eigenvalues': eigenvalues,
        'training_filenames': filenames,
        'person_name': person_name,
        'version': version,
        'training_timestamp': datetime.now().isoformat(),
        'n_components': eigenfaces.shape[1],
        'face_dimensions': eigenfaces.shape[0]
    }
    
    # Save model using pickle with version suffix
    if version:
        model_filename = f"{person_name}_{version}_pca_model.pkl"
        metadata_filename = f"{person_name}_{version}_model_info.json"
    else:
        model_filename = f"{person_name}_pca_model.pkl"
        metadata_filename = f"{person_name}_model_info.json"
    
    model_path = os.path.join(model_dir, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"PCA model saved to: {model_path}")
    
    # Save model metadata as JSON for easy inspection
    metadata = {
        'person_name': person_name,
        'version': version,
        'training_timestamp': model_data['training_timestamp'],
        'n_components': model_data['n_components'],
        'face_dimensions': model_data['face_dimensions'],
        'n_training_images': len(filenames),
        'explained_variance_ratio': (eigenvalues / np.sum(eigenvalues)).tolist()[:10],
        'model_file': model_filename
    }
    
    metadata_path = os.path.join(model_dir, metadata_filename)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Model metadata saved to: {metadata_path}")
    
    return model_path

def visualize_eigenfaces(eigenfaces, mean_face, output_dir, person_name, n_display=10):
    """
    Save visualizations of eigenfaces and mean face
    
    Args:
        eigenfaces (np.array): Principal component vectors
        mean_face (np.array): Mean face vector
        output_dir (str): Directory to save visualizations
        person_name (str): Name of the person
        n_display (int): Number of eigenfaces to visualize
    """
    # Determine face image dimensions (assuming square faces)
    face_dim = int(np.sqrt(len(mean_face)))
    
    # Save mean face
    mean_face_img = mean_face.reshape(face_dim, face_dim)
    mean_face_normalized = cv2.normalize(mean_face_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mean_face_path = os.path.join(output_dir, f"{person_name}_mean_face.jpg")
    cv2.imwrite(mean_face_path, mean_face_normalized)
    print(f"Mean face saved to: {mean_face_path}")
    
    # Save top eigenfaces
    n_display = min(n_display, eigenfaces.shape[1])
    for i in range(n_display):
        eigenface = eigenfaces[:, i].reshape(face_dim, face_dim)
        eigenface_normalized = cv2.normalize(eigenface, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        eigenface_path = os.path.join(output_dir, f"{person_name}_eigenface_{i+1:02d}.jpg")
        cv2.imwrite(eigenface_path, eigenface_normalized)
    
    print(f"Saved {n_display} eigenfaces to: {output_dir}")

def train_single_model(faces_dir, person_name, model_dir, version, n_components=50):
    """
    Train a single PCA model for a specific version
    
    Args:
        faces_dir (str): Directory containing face images
        person_name (str): Name of the person
        model_dir (str): Directory to save model files
        version (str): Version identifier (e.g., 'dark', 'light')
        n_components (int): Number of principal components to keep
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\n=== Training {version.upper()} version model ===")
        
        # Check if faces directory exists
        if not os.path.exists(faces_dir):
            raise FileNotFoundError(f"Faces directory not found: {faces_dir}")
        
        # Load face images
        face_matrix, filenames = load_face_images(faces_dir)
        
        # Perform PCA
        eigenfaces, mean_face, projected_data, eigenvalues = manual_pca(
            face_matrix, n_components=n_components
        )
        
        # Save the trained model with version
        model_path = save_pca_model(
            eigenfaces, mean_face, projected_data, eigenvalues, 
            filenames, person_name, model_dir, version
        )
        
        # Save eigenface visualizations with version
        visualize_eigenfaces(eigenfaces, mean_face, model_dir, f"{person_name}_{version}")
        
        print(f"\n=== {version.upper()} PCA Training Summary ===")
        print(f"Person: {person_name}")
        print(f"Version: {version}")
        print(f"Training images: {len(filenames)}")
        print(f"Principal components: {eigenfaces.shape[1]}")
        print(f"Face dimensions: {eigenfaces.shape[0]}")
        print(f"Model saved to: {model_path}")
        print(f"Total explained variance: {np.sum(eigenvalues / np.sum(eigenvalues)) * 100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"Error during {version} PCA training: {str(e)}")
        return False

def main():
    """
    Main function to train PCA models for both Dark and Light versions
    """
    # Configuration
    base_faces_dir = "faces"
    model_dir = "models"
    person_name = "Joseph_Lai"
    n_components = 50  # Number of principal components to keep
    
    # Define the two versions to train
    versions = [
        {"name": "dark", "dir": os.path.join(base_faces_dir, "Dark_version")},
        {"name": "light", "dir": os.path.join(base_faces_dir, "Light_version")}
    ]
    
    success_count = 0
    
    print("=== Starting PCA Training for Multiple Versions ===")
    print(f"Person: {person_name}")
    print(f"Versions to train: {[v['name'] for v in versions]}")
    
    # Train each version
    for version_info in versions:
        version_name = version_info["name"]
        version_dir = version_info["dir"]
        
        print(f"\n--- Processing {version_name.upper()} version ---")
        print(f"Looking for images in: {version_dir}")
        
        if train_single_model(version_dir, person_name, model_dir, version_name, n_components):
            success_count += 1
        else:
            print(f"Failed to train {version_name} model")
    
    # Final summary
    print("\n" + "="*50)
    print("=== FINAL TRAINING SUMMARY ===")
    print(f"Total models trained successfully: {success_count}/{len(versions)}")
    
    if success_count == len(versions):
        print("✅ All models trained successfully!")
        print(f"Models saved:")
        for version_info in versions:
            version_name = version_info["name"]
            model_file = f"{person_name}_{version_name}_pca_model.pkl"
            print(f"  - {model_file}")
        return True
    else:
        print("❌ Some models failed to train")
        return False

if __name__ == "__main__":
    main()