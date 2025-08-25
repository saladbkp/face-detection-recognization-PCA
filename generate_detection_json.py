import cv2
import json
import os
import glob
import re
from datetime import datetime

def generate_detection_json_for_person(person_name, face_dir):
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
    
    print(f"Found {len(face_images)} face images")
    
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

def main():
    """
    Generate detection JSON files for all persons in lock_version directory
    """
    base_dir = "faces/lock_version"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found!")
        return
    
    # Get all person directories
    person_dirs = [d for d in os.listdir(base_dir) 
                   if os.path.isdir(os.path.join(base_dir, d))]
    
    print(f"Found {len(person_dirs)} person directories: {person_dirs}")
    
    generated_files = []
    
    for person_name in person_dirs:
        person_dir = os.path.join(base_dir, person_name)
        
        # Check if JSON file already exists
        json_path = os.path.join(person_dir, f"{person_name}_faces_detection.json")
        
        if os.path.exists(json_path):
            print(f"JSON file already exists for {person_name}: {json_path}")
            continue
        
        # Generate JSON file
        result = generate_detection_json_for_person(person_name, person_dir)
        if result:
            generated_files.append(result)
    
    print(f"\nGeneration completed!")
    print(f"Generated {len(generated_files)} JSON files:")
    for file_path in generated_files:
        print(f"  - {file_path}")

if __name__ == "__main__":
    main()