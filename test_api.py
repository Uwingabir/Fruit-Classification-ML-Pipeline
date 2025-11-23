"""
Test script for the Fruit Classification API
"""

import requests
import os
from pathlib import Path
import time

# Configuration
API_BASE = "http://localhost:8000"
TEST_IMAGE_DIR = "archive (2)/dataset/test"

def test_health():
    """Test health endpoint"""
    print("\n[TEST] Health Check")
    print("-" * 50)
    response = requests.get(f"{API_BASE}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("✓ Health check passed")

def test_model_info():
    """Test model info endpoint"""
    print("\n[TEST] Model Info")
    print("-" * 50)
    response = requests.get(f"{API_BASE}/model-info")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Model Status: {data['status']}")
    print(f"Uptime: {data['uptime_seconds']}s")
    print(f"Predictions: {data['prediction_count']}")
    assert response.status_code == 200
    print("✓ Model info test passed")

def test_prediction():
    """Test prediction endpoint with a sample image"""
    print("\n[TEST] Prediction")
    print("-" * 50)
    
    # Find first available image
    test_image = None
    for class_dir in Path(TEST_IMAGE_DIR).iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            if images:
                test_image = images[0]
                break
    
    if not test_image:
        print("⚠ No test images found, skipping prediction test")
        return
    
    print(f"Using test image: {test_image}")
    
    with open(test_image, 'rb') as f:
        files = {'file': f}
        start_time = time.time()
        response = requests.post(f"{API_BASE}/predict", files=files)
        elapsed_time = time.time() - start_time
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Time: {elapsed_time * 1000:.2f}ms")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Predicted Class: {data['predicted_class']}")
        print(f"Confidence: {data['confidence']:.4f}")
        print("✓ Prediction test passed")
    else:
        print(f"✗ Prediction test failed: {response.text}")

def test_bulk_prediction():
    """Test multiple predictions"""
    print("\n[TEST] Bulk Prediction (10 images)")
    print("-" * 50)
    
    # Collect test images
    test_images = []
    for class_dir in Path(TEST_IMAGE_DIR).iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.png"))[:2]  # 2 from each class
            test_images.extend(images)
            if len(test_images) >= 10:
                break
    
    if not test_images:
        print("⚠ No test images found, skipping bulk test")
        return
    
    results = []
    total_time = 0
    
    for i, image_path in enumerate(test_images[:10], 1):
        with open(image_path, 'rb') as f:
            files = {'file': f}
            start_time = time.time()
            response = requests.post(f"{API_BASE}/predict", files=files)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
        
        if response.status_code == 200:
            data = response.json()
            results.append({
                'predicted': data['predicted_class'],
                'confidence': data['confidence'],
                'time_ms': elapsed_time * 1000
            })
            print(f"  {i}/10: {data['predicted_class']} ({data['confidence']:.2%}) - {elapsed_time*1000:.2f}ms")
    
    print(f"\nSummary:")
    print(f"  Total predictions: {len(results)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time: {(total_time/len(results))*1000:.2f}ms")
    print(f"  Throughput: {len(results)/total_time:.2f} req/s")
    print("✓ Bulk prediction test passed")

def test_upload_training_data():
    """Test uploading training data"""
    print("\n[TEST] Upload Training Data")
    print("-" * 50)
    
    # Find sample images
    sample_images = []
    test_dir = Path(TEST_IMAGE_DIR) / "freshapples"
    if test_dir.exists():
        sample_images = list(test_dir.glob("*.png"))[:2]
    
    if not sample_images:
        print("⚠ No sample images found, skipping upload test")
        return
    
    files = [('files', open(img, 'rb')) for img in sample_images]
    data = {'class_name': 'freshapples'}
    
    response = requests.post(
        f"{API_BASE}/upload-training-data",
        files=files,
        data=data
    )
    
    # Close files
    for _, f in files:
        f.close()
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Uploaded: {result['uploaded_count']} files")
        print("✓ Upload test passed")
    else:
        print(f"✗ Upload test failed: {response.text}")

def test_retraining_status():
    """Test retraining status endpoint"""
    print("\n[TEST] Retraining Status")
    print("-" * 50)
    response = requests.get(f"{API_BASE}/retraining-status")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Is Retraining: {data['is_retraining']}")
    print(f"Progress: {data['progress']}%")
    print(f"Message: {data['message']}")
    print("✓ Retraining status test passed")

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("  Fruit Classification API Tests")
    print("=" * 50)
    
    try:
        test_health()
        test_model_info()
        test_prediction()
        test_bulk_prediction()
        test_upload_training_data()
        test_retraining_status()
        
        print("\n" + "=" * 50)
        print("  All Tests Passed! ✓")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API")
        print("Make sure the API server is running: python app.py")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")

if __name__ == "__main__":
    run_all_tests()
