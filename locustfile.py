import os
import random
import time
from locust import HttpUser, task, between, events
from io import BytesIO
from PIL import Image
import numpy as np


class FruitClassifierUser(HttpUser):
    """
    Locust user class for load testing the Fruit Classification API.
    Simulates users uploading images for prediction.
    """
    
    # Wait time between tasks (1-3 seconds)
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a simulated user starts."""
        self.test_images = self.generate_test_images()
        print(f"User started with {len(self.test_images)} test images")
    
    def generate_test_images(self, count=5):
        """
        Generate synthetic test images for load testing.
        
        Args:
            count (int): Number of images to generate
            
        Returns:
            list: List of image bytes
        """
        images = []
        
        # Generate random colored images
        colors = [
            (255, 0, 0),    # Red (apple-like)
            (255, 255, 0),  # Yellow (banana-like)
            (255, 165, 0),  # Orange (orange-like)
            (150, 50, 50),  # Dark red (rotten apple)
            (100, 100, 0),  # Dark yellow (rotten banana)
            (150, 100, 0),  # Dark orange (rotten orange)
        ]
        
        for i in range(count):
            # Create a random image
            color = random.choice(colors)
            img = Image.new('RGB', (224, 224), color)
            
            # Add some noise
            pixels = np.array(img)
            noise = np.random.randint(-30, 30, pixels.shape, dtype=np.int16)
            pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(pixels)
            
            # Convert to bytes
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            images.append(img_byte_arr.getvalue())
        
        return images
    
    @task(10)
    def predict_image(self):
        """
        Task: Send a prediction request.
        Weight: 10 (most common task)
        """
        # Select a random test image
        image_bytes = random.choice(self.test_images)
        
        # Prepare the file for upload
        files = {
            'file': ('test_image.png', BytesIO(image_bytes), 'image/png')
        }
        
        # Send prediction request
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    json_data = response.json()
                    if 'predicted_class' in json_data:
                        response.success()
                    else:
                        response.failure("Missing predicted_class in response")
                except Exception as e:
                    response.failure(f"Failed to parse JSON: {str(e)}")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def check_health(self):
        """
        Task: Check API health.
        Weight: 2 (occasional task)
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(1)
    def get_model_info(self):
        """
        Task: Get model information.
        Weight: 1 (rare task)
        """
        with self.client.get("/model-info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Model info request failed with status {response.status_code}")


# Custom statistics tracking
request_times = []
error_count = 0
success_count = 0


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track request statistics."""
    global request_times, error_count, success_count
    
    if exception:
        error_count += 1
    else:
        success_count += 1
        request_times.append(response_time)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print summary statistics when test stops."""
    global request_times, error_count, success_count
    
    if request_times:
        print("\n" + "="*50)
        print("LOAD TEST SUMMARY")
        print("="*50)
        print(f"Total Requests: {success_count + error_count}")
        print(f"Successful: {success_count}")
        print(f"Failed: {error_count}")
        print(f"Success Rate: {(success_count/(success_count+error_count)*100):.2f}%")
        print(f"\nResponse Time Statistics (ms):")
        print(f"  Min: {min(request_times):.2f}")
        print(f"  Max: {max(request_times):.2f}")
        print(f"  Mean: {np.mean(request_times):.2f}")
        print(f"  Median: {np.median(request_times):.2f}")
        print(f"  95th Percentile: {np.percentile(request_times, 95):.2f}")
        print(f"  99th Percentile: {np.percentile(request_times, 99):.2f}")
        print("="*50 + "\n")


# For running standalone (not typical, but useful for testing)
if __name__ == "__main__":
    import subprocess
    import sys
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         Fruit Classification API - Load Tester            ║
    ╚════════════════════════════════════════════════════════════╝
    
    Usage:
    ------
    Basic test (Web UI):
        locust -f locustfile.py --host=http://localhost:8000
    
    Headless test (1000 users, 100 spawn rate, 60 seconds):
        locust -f locustfile.py --host=http://localhost:8000 \\
               --users 1000 --spawn-rate 100 --run-time 60s --headless
    
    Different scenarios:
    --------------------
    1. Light load:
        locust -f locustfile.py --host=http://localhost:8000 \\
               --users 10 --spawn-rate 2 --run-time 60s --headless
    
    2. Medium load:
        locust -f locustfile.py --host=http://localhost:8000 \\
               --users 100 --spawn-rate 10 --run-time 120s --headless
    
    3. Heavy load:
        locust -f locustfile.py --host=http://localhost:8000 \\
               --users 500 --spawn-rate 50 --run-time 180s --headless
    
    4. Stress test:
        locust -f locustfile.py --host=http://localhost:8000 \\
               --users 2000 --spawn-rate 200 --run-time 300s --headless
    
    View results:
    -------------
    - Web UI: http://localhost:8089 (when running without --headless)
    - CSV export: Add --csv=results to save results
    - HTML report: Add --html=report.html to generate HTML report
    """)
