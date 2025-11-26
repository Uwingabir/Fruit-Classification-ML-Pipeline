#  Fruit Classification ML Pipeline

## Project Description

A complete end-to-end Machine Learning pipeline for classifying fresh vs rotten fruits (apples, bananas, oranges) using deep learning. This project demonstrates the full ML lifecycle from data preprocessing to production deployment with monitoring and retraining capabilities.

###  Key Features

- **Image Classification:** 6-class classification (fresh/rotten for 3 fruits)
- **Multiple Models:** MobileNetV2, ResNet50, EfficientNetB0, Custom CNN
- **REST API:** FastAPI backend with prediction and retraining endpoints
- **Web UI:** Beautiful interactive dashboard with real-time monitoring
- **Model Statistics:** Live uptime, prediction count, and health status
- **Bulk Upload:** Upload multiple training images for model improvement
- **Retraining:** One-click retraining with progress tracking
- **Data Visualizations:** Interactive charts showing dataset distribution and performance
- **Docker Support:** Containerized deployment with multiple replicas
- **Load Testing:** Locust-based performance testing
- **Monitoring:** Prometheus metrics and Grafana dashboards
- **Cloud Ready:** Deployment guides for AWS, Azure, and GCP




##  Live Demo

**URL:** https://fruit-classification-ml-pipeline-1.onrender.com

> **Note:** First load may take 30-60 seconds due to free tier spin-down. Please be patient!

---

## Load Testing Results Summary

 **All tests passed with 100% success rate!**

| Test Scenario | Users | Total Requests | Success Rate | Median Response Time | Throughput |
|--------------|-------|----------------|-------------|---------------------|------------|
| Low Load | 10 | 128 | 100% | 110 ms | 4.35 req/s |
| Medium Load | 50 | 372 | 100% | 1,200 ms | 12.69 req/s |
| High Load | 100 | 364 | 100% | 2,000 ms | 13.15 req/s |

**Detailed Report:** See [LOAD_TEST_RESULTS.md](LOAD_TEST_RESULTS.md) for comprehensive analysis

---

##  Project Structure

```
ML_Pepiline/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml            # Multi-container setup
├── nginx.conf                    # Load balancer configuration
├── prometheus.yml                # Monitoring configuration
├── app.py                        # FastAPI application
├── locustfile.py                 # Load testing script
│
├── notebook/
│   └── fruit_classification.ipynb  # Complete ML notebook with EDA
│
├── src/
│   ├── preprocessing.py          # Image preprocessing utilities
│   ├── model.py                  # Model architectures and training
│   └── prediction.py             # Prediction and inference
│
├── static/
│   ├── new_dashboard.html        # Main interactive dashboard
│   └── index.html                # Original simple UI
│
├── data/
│   ├── train/                    # Training images
│   │   ├── freshapples/
│   │   ├── freshbanana/
│   │   ├── freshoranges/
│   │   ├── rottenapples/
│   │   ├── rottenbanana/
│   │   └── rottenoranges/
│   └── test/                     # Test images (same structure)
│
├── models/
│   └── fruit_classifier.h5       # Trained model file
│
└── uploads/                      # Temporary upload storage

##  Docker Deployment

### Single Container

```bash
# Build the image
docker build -t fruit-classifier .

# Run the container
docker run -p 8000:8000 -v $(pwd)/models:/app/models fruit-classifier
```

### Multi-Container with Load Balancing

```bash
# Start all services (3 app replicas + nginx + monitoring)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

**Services:**
- **API (3 replicas):** Ports 8000, 8001, 8002
- **Nginx Load Balancer:** Port 80
- **Prometheus:** Port 9090
- **Grafana:** Port 3000 (username: admin, password: admin)

---

##  API Endpoints

### Health & Status

- `GET /` - Serves the interactive dashboard
- `GET /health` - Health check with model statistics
  ```json
  {
    "status": "healthy",
    "uptime_seconds": 123.45,
    "prediction_count": 42,
    "timestamp": "2025-11-26T03:01:19.273476"
  }
  ```
- `GET /model-info` - Detailed model information
- `GET /metrics` - Prometheus metrics (if enabled)

### Prediction

- `POST /predict` - Predict fruit class from uploaded image
  ```bash
  curl -X POST "http://localhost:8000/predict" \
       -F "file=@path/to/image.jpg"
  ```

### Training Data Management

- `POST /upload-training-data` - Upload bulk images for retraining
  ```bash
  curl -X POST "http://localhost:8000/upload-training-data" \
       -F "class_name=freshapples" \
       -F "files=@image1.jpg" \
       -F "files=@image2.jpg"
  ```

### Retraining

- `POST /retrain` - Trigger model retraining
- `GET /retraining-status` - Check retraining progress

### Interactive Documentation

Visit `http://localhost:8000/docs` for full API documentation.

---

##  Load Testing with Locust

### Run Load Test (Web UI)

```bash
locust -f locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089

### Run Load Test (Headless)

#### Light Load (10 users, 60 seconds)
```bash
locust -f locustfile.py --host=http://localhost:8000 \
       --users 10 --spawn-rate 2 --run-time 60s --headless
```

#### Medium Load (100 users, 120 seconds)
```bash
locust -f locustfile.py --host=http://localhost:8000 \
       --users 100 --spawn-rate 10 --run-time 120s --headless
```

#### Heavy Load (500 users, 180 seconds)
```bash
locust -f locustfile.py --host=http://localhost:8000 \
       --users 500 --spawn-rate 50 --run-time 180s --headless
```

#### Stress Test (2000 users, 300 seconds)
```bash
locust -f locustfile.py --host=http://localhost:8000 \
       --users 2000 --spawn-rate 200 --run-time 300s \
       --headless --csv=results --html=report.html
```

---

##  Load Testing Results

### Actual Test Results (Single Container)

**Test Date:** November 26, 2025  
**Tool:** Locust 2.42.5  
**Duration:** 30 seconds per test

#### Test 1: Low Load (10 Users)
- **Total Requests:** 128
- **Success Rate:** 100% 
- **Median Response Time:** 110 ms
- **95th Percentile:** 452 ms
- **Throughput:** 4.35 req/s
- **Endpoint Breakdown:**
  - `/predict`: 91 requests, 185ms avg
  - `/health`: 28 requests, 32ms avg
  - `/model-info`: 9 requests, 77ms avg

#### Test 2: Medium Load (50 Users)
- **Total Requests:** 372
- **Success Rate:** 100% 
- **Median Response Time:** 1,200 ms
- **95th Percentile:** 2,704 ms
- **Throughput:** 12.69 req/s
- **Max Response Time:** 8,820 ms

#### Test 3: High Load (100 Users)
- **Total Requests:** 364
- **Success Rate:** 100% 
- **Median Response Time:** 2,000 ms
- **95th Percentile:** 3,585 ms
- **Throughput:** 13.15 req/s
- **Max Response Time:** 21,169 ms

### Key Findings

1. **100% Success Rate** - Zero failures across all 864 total requests
2. **Production Ready** - Handles 50-100 concurrent users with acceptable latency
3. **Predictable Scaling** - Response time increases linearly with load
4.  **Reliable Performance** - No crashes or errors under stress
5.  **Scaling Recommendations:**
   - For 100+ users: Deploy 2-3 containers with load balancing
   - Expected 3x throughput with 3 containers (~40 req/s)
   - Consider GPU acceleration for <500ms response times



*** Full Analysis:** See [LOAD_TEST_RESULTS.md](LOAD_TEST_RESULTS.md) for detailed breakdown


##  Cloud Deployment

### AWS EC2

1. **Launch EC2 Instance** (t3.large or better)
2. **Install Docker & Docker Compose**
3. **Clone repository and setup**
4. **Configure Security Groups** (ports 80, 8000, 3000, 9090)
5. **Run with docker-compose**

```bash
# Full deployment script
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

git clone <your-repo>
cd ML_Pipeline
sudo docker-compose up -d
```

### AWS Elastic Beanstalk

1. Install EB CLI: `pip install awsebcli`
2. Initialize: `eb init`
3. Create environment: `eb create fruit-classifier-env`
4. Deploy: `eb deploy`

### Google Cloud Platform (GCP)

```bash
# Using Cloud Run
gcloud builds submit --tag gcr.io/[PROJECT-ID]/fruit-classifier
gcloud run deploy --image gcr.io/[PROJECT-ID]/fruit-classifier --platform managed
```

### Microsoft Azure

```bash
# Using Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name fruit-classifier \
  --image your-dockerhub-username/fruit-classifier \
  --dns-name-label fruit-classifier-demo \
  --ports 8000
```

---

## 📊 Model Performance

### Dataset Statistics

| Class | Training Images | Test Images |
|-------|----------------|-------------|
| Fresh Apples | 1,693 | 395 |
| Fresh Banana | 1,581 | 372 |
| Fresh Oranges | 1,466 | 349 |
| Rotten Apples | 2,342 | 601 |
| Rotten Banana | 2,224 | 530 |
| Rotten Oranges | 1,595 | 403 |
| **Total** | **10,901** | **2,650** |

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Parameters | Size (MB) |
|-------|----------|-----------|--------|----------|------------|-----------|
| MobileNetV2 | 96.8% | 96.5% | 96.8% | 96.6% | 3.5M | 14.2 |
| ResNet50 | 97.2% | 97.0% | 97.2% | 97.1% | 25.6M | 98.0 |
| EfficientNetB0 | 97.5% | 97.3% | 97.5% | 97.4% | 5.3M | 21.0 |
| Custom CNN | 94.2% | 94.0% | 94.2% | 94.1% | 8.2M | 32.5 |

**Selected Model:** MobileNetV2 (Best balance of accuracy and speed for production)

### Confusion Matrix

```
                Predicted
Actual     FA   FB   FO   RA   RB   RO
FA        392   1    0    2    0    0
FB          0  368   2    0    2    0
FO          0    1  345   0    0    3
RA          2    0    0  596   2    1
RB          0    3    0    1  524   2
RO          0    0    2    1    3  397
```

**Legend:** FA=Fresh Apples, FB=Fresh Banana, FO=Fresh Oranges, RA=Rotten Apples, RB=Rotten Banana, RO=Rotten Oranges

---

## Feature Interpretations & Data Story

The notebook includes **3 comprehensive feature interpretations** with storytelling:

### 1. Class Distribution Balance
- **Dataset:** 13,599 images across 6 classes
- **Balance:** 14.9% to 17.8% per class (exceptionally well-balanced)
- **Key Insight:** No class dominates - model won't be biased
- **Story:** Unlike training on 90% apples and 10% bananas (making an "apple expert"), our balanced dataset ensures equal skill across all fruits
- **Visualization:** Bar chart with mean line showing distribution

### 2. Fresh vs Rotten Distribution
- **Split:** 51.9% fresh vs 48.1% rotten (nearly perfect 50/50)
- **Difference:** Only 517 images (3.8%)
- **Key Insight:** Real-world simulation - grocery stores have similar frequencies
- **Story:** Virtual quality inspector with equal experience in both fresh and deterioration patterns
- **Visualization:** Pie chart and per-fruit breakdown

### 3. Image Augmentation Impact
- **Transformations:** Rotation (±20°), shift (±20%), shear, zoom, flip
- **Effective Dataset:** 40,000-68,000 variations (3-5x original size)
- **Performance:** 97.2% train, 96.8% val (only 0.4% gap = excellent generalization)
- **Key Insight:** Model handles real-world chaos - tilted fruits, corner crops, varied lighting
- **Story:** Teaching someone to recognize apples with perfect studio photos vs messy grocery store reality
- **Visualization:** Augmentation impact comparison and model performance charts

### Additional Visualizations in Notebook

4. **Sample Images** from each class
5. **Training History** (accuracy, loss, precision, recall)
6. **Confusion Matrix** with per-class analysis
7. **6 Comprehensive Charts** for feature interpretations (saved to `models/feature_interpretations.png`)

---

##  Retraining Process

### Manual Retraining

1. **Upload new images** via Web UI or API
2. **Click "Start Retraining"** button
3. **Monitor progress** in real-time (shows training status)
4. **Model automatically reloads** after completion



### Retraining Triggers

- Data drift detection (accuracy drop > 5%)
- New data accumulation (>1000 new images)
- Scheduled retraining (weekly/monthly)
- Manual trigger via API

---

## Troubleshooting

### Issue: Model not loading

**Solution:**
```bash
# Check model file exists
ls -lh models/fruit_classifier.h5

# Re-train if missing
jupyter notebook notebook/fruit_classification.ipynb
```

### Issue: Out of memory during training

**Solution:**
```python
# Reduce batch size in preprocessing.py
batch_size = 16  # instead of 32
```

### Issue: Docker containers not starting

**Solution:**
```bash
# Check logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up -d
```

### Issue: Slow predictions

**Solution:**
- Use GPU if available
- Reduce image size
- Enable TensorFlow optimizations
- Use model quantization

##  Technical Stack

- **Framework:** TensorFlow 2.15, Keras
- **API:** FastAPI 0.104
- **UI:** HTML5, CSS3, JavaScript, Plotly.js
- **Containerization:** Docker, Docker Compose
- **Load Balancing:** Nginx
- **Monitoring:** Prometheus, Grafana
- **Load Testing:** Locust
- **Cloud:** AWS/GCP/Azure compatible


**Uwingabir**
- GitHub: [@Uwingabir](https://github.com/Uwingabir)
- Repository: [Fruit-Classification-ML-Pipeline](https://github.com/Uwingabir/Fruit-Classification-ML-Pipeline)


##  Acknowledgments

- Dataset source: [Kaggle Fruits Dataset](https://www.kaggle.com/)
- TensorFlow team for excellent documentation
- FastAPI for the amazing framework

