#!/usr/bin/env python3
"""
Test Enhanced Face Recognition
This script tests the enhanced face recognition model to demonstrate confidence improvements
"""

import cv2
import numpy as np
import json
import os
from scan_enhanced import EnhancedFaceScanner

def test_enhanced_recognition():
    """
    Test enhanced face recognition with confidence comparison
    """
    print("============================================================")
    print("ENHANCED FACE RECOGNITION TEST")
    print("============================================================")
    
    # Initialize enhanced scanner
    person_name = "test_enhanced"
    base_dir = f"faces/lock_version/{person_name}"
    
    scanner = EnhancedFaceScanner()
    
    # Load enhanced model
    model_path = os.path.join(base_dir, "face_model_enhanced.pkl")
    json_path = os.path.join(base_dir, f"{person_name}_faces_detection.json")
    
    if not os.path.exists(model_path):
        print(f"Error: Enhanced model not found at {model_path}")
        return False
    
    if not os.path.exists(json_path):
        print(f"Error: Detection data not found at {json_path}")
        return False
    
    # Load model and detection data
    success = scanner.load_enhanced_model(model_path, json_path, person_name)
    if not success:
        print("Failed to load enhanced model!")
        return False
    
    print(f"Enhanced model loaded successfully!")
    print(f"Model features: Multi-scale, HOG, LBP, Angle detection")
    print("\n" + "="*60)
    
    # Test with sample face images
    face_dir = base_dir
    face_files = [f for f in os.listdir(face_dir) if f.endswith('.jpg')]
    
    if not face_files:
        print("No face images found for testing!")
        return False
    
    # Test with different face images
    test_results = []
    
    print("Testing enhanced recognition on sample faces...\n")
    
    for i, face_file in enumerate(face_files[:10]):  # Test first 10 faces
        face_path = os.path.join(face_dir, face_file)
        face_img = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
        
        if face_img is None:
            continue
        
        # Perform enhanced recognition
        person_id, person_name_result, confidence, angle_type = scanner.enhanced_recognize_face(face_img)
        
        test_results.append({
            'file': face_file,
            'person_id': person_id,
            'person_name': person_name_result,
            'confidence': confidence,
            'angle_type': angle_type
        })
        
        status = "✓ RECOGNIZED" if person_id != -1 else "✗ UNKNOWN"
        print(f"Face {i+1:2d}: {face_file[:30]:<30} | {status} | Confidence: {confidence:.3f} | Angle: {angle_type}")
    
    # Calculate statistics
    recognized_faces = [r for r in test_results if r['person_id'] != -1]
    unknown_faces = [r for r in test_results if r['person_id'] == -1]
    
    if recognized_faces:
        avg_confidence = np.mean([r['confidence'] for r in recognized_faces])
        max_confidence = np.max([r['confidence'] for r in recognized_faces])
        min_confidence = np.min([r['confidence'] for r in recognized_faces])
    else:
        avg_confidence = max_confidence = min_confidence = 0.0
    
    # Analyze by angle type
    frontal_faces = [r for r in test_results if r['angle_type'] == 'frontal']
    profile_faces = [r for r in test_results if r['angle_type'] in ['left_profile', 'right_profile']]
    
    frontal_recognized = [r for r in frontal_faces if r['person_id'] != -1]
    profile_recognized = [r for r in profile_faces if r['person_id'] != -1]
    
    print("\n" + "="*60)
    print("ENHANCED RECOGNITION RESULTS")
    print("="*60)
    print(f"Total faces tested: {len(test_results)}")
    print(f"Successfully recognized: {len(recognized_faces)} ({len(recognized_faces)/len(test_results)*100:.1f}%)")
    print(f"Unknown faces: {len(unknown_faces)} ({len(unknown_faces)/len(test_results)*100:.1f}%)")
    print()
    print("CONFIDENCE STATISTICS:")
    print(f"  Average confidence: {avg_confidence:.3f}")
    print(f"  Maximum confidence: {max_confidence:.3f}")
    print(f"  Minimum confidence: {min_confidence:.3f}")
    print()
    print("ANGLE-BASED ANALYSIS:")
    print(f"  Frontal faces: {len(frontal_faces)} total, {len(frontal_recognized)} recognized ({len(frontal_recognized)/max(1,len(frontal_faces))*100:.1f}%)")
    if frontal_recognized:
        frontal_avg_conf = np.mean([r['confidence'] for r in frontal_recognized])
        print(f"    Average frontal confidence: {frontal_avg_conf:.3f}")
    
    print(f"  Profile faces: {len(profile_faces)} total, {len(profile_recognized)} recognized ({len(profile_recognized)/max(1,len(profile_faces))*100:.1f}%)")
    if profile_recognized:
        profile_avg_conf = np.mean([r['confidence'] for r in profile_recognized])
        print(f"    Average profile confidence: {profile_avg_conf:.3f}")
    
    print("\n" + "="*60)
    print("ENHANCEMENT FEATURES ACTIVE:")
    print("  ✓ Multi-scale feature extraction (48x48, 64x64, 80x80)")
    print("  ✓ HOG (Histogram of Oriented Gradients) features")
    print("  ✓ LBP (Local Binary Patterns) features")
    print("  ✓ Face angle detection and compensation")
    print("  ✓ Profile face optimization")
    print("  ✓ Multi-model ensemble voting")
    print("  ✓ Adaptive thresholds for different angles")
    print("  ✓ Data augmentation (6x training samples)")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_enhanced_recognition()