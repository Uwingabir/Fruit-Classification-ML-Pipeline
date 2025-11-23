# 🚀 Quick Start Guide

## Overview

This guide will help you get the Fruit Classification ML Pipeline up and running in **under 10 minutes**.

---

## Prerequisites Checklist

- [ ] Python 3.10 or higher installed
- [ ] pip package manager
- [ ] 4GB+ RAM available
- [ ] 2GB+ disk space
- [ ] (Optional) Docker for containerized deployment
- [ ] (Optional) GPU for faster training

---

## Step 1: Setup Environment (2 minutes)

### Automatic Setup

```bash
# Run the automated setup script
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Create necessary directories
- Verify dataset structure

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p models uploads data/retrain logs
```

---

## Step 2: Prepare Your Dataset (1 minute)

Ensure your dataset is organized as:

```
archive (2)/dataset/
├── train/
│   ├── freshapples/     (images here)
│   ├── freshbanana/     (images here)
│   ├── freshoranges/    (images here)
│   ├── rottenapples/    (images here)
│   ├── rottenbanana/    (images here)
│   └── rottenoranges/   (images here)
└── test/
    └── (same structure)
```

**Quick Check:**
```bash
ls "archive (2)/dataset/train"
```

You should see 6 directories.

---

## Step 3: Train the Model (3-5 minutes)

### Option A: Using Jupyter Notebook (Recommended)

```bash
jupyter notebook notebook/fruit_classification.ipynb
```

Then:
1. Run all cells (Cell → Run All)
2. Wait for training to complete (~5 minutes)
3. Model will be saved to `models/fruit_classifier.h5`

### Option B: Using Training Script

```bash
./train_model.sh
```

This will automatically:
- Load and preprocess data
- Train a MobileNetV2 model
- Evaluate on test set
- Save the model

---

## Step 4: Run the API (30 seconds)

```bash
python app.py
```

The API will start at: http://localhost:8000

**Quick Test:**
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "healthy",
  "uptime_seconds": 5.2,
  "prediction_count": 0
}
```

---

## Step 5: Make Your First Prediction (1 minute)

### Using the Web UI

1. Open http://localhost:8000 in your browser
2. Click "Choose File" and select a fruit image
3. Click "Predict"
4. See the results!

### Using API

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@path/to/your/fruit_image.jpg"
```

Expected output:
```json
{
  "predicted_class": "freshapples",
  "confidence": 0.9823,
  "probabilities": {...},
  "interpretation": "This appears to be a fresh apples..."
}
```

---

## Step 6: Test with Load (Optional, 2 minutes)

### Start Locust

```bash
locust -f locustfile.py --host=http://localhost:8000
```

### Run a Quick Test

Open http://localhost:8089 and:
1. Set users: **10**
2. Set spawn rate: **2**
3. Click "Start"
4. Watch the metrics!

---

## Step 7: Deploy with Docker (Optional, 3 minutes)

### Single Container

```bash
docker build -t fruit-classifier .
docker run -p 8000:8000 fruit-classifier
```

### Multi-Container with Load Balancing

```bash
docker-compose up -d
```

This starts:
- 3 API replicas
- Nginx load balancer
- Prometheus monitoring
- Grafana dashboards

**Access Points:**
- API: http://localhost:8000
- Load Balanced: http://localhost
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

---

## Troubleshooting

### "Model not found" error

```bash
# Check if model exists
ls -lh models/fruit_classifier.h5

# If missing, train the model
./train_model.sh
```

### "Dataset not found" error

```bash
# Verify dataset structure
ls "archive (2)/dataset/train"

# Should show 6 directories
```

### API not starting

```bash
# Check if port 8000 is available
lsof -i :8000

# Kill any process using the port
kill -9 <PID>
```

### Slow predictions

- Enable GPU acceleration (if available)
- Reduce image size in `preprocessing.py`
- Use a smaller model (MobileNetV2 instead of ResNet50)

---

## Common Commands

### Check API Status
```bash
curl http://localhost:8000/health
```

### Get Model Info
```bash
curl http://localhost:8000/model-info
```

### Run Tests
```bash
python test_api.py
```

### View Logs
```bash
# Docker logs
docker-compose logs -f

# Application logs
tail -f logs/app.log
```

### Stop Everything
```bash
# Stop API
Ctrl+C

# Stop Docker
docker-compose down
```

---

## Next Steps

1. ✅ **Explore the Notebook:** Review detailed analysis and visualizations
2. ✅ **Try the Web UI:** Use the interactive dashboard
3. ✅ **Run Load Tests:** Test with different user loads
4. ✅ **Deploy to Cloud:** Use AWS/GCP/Azure deployment scripts
5. ✅ **Customize Model:** Experiment with different architectures
6. ✅ **Add More Data:** Upload new images and retrain

---

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API homepage |
| `/health` | GET | Health check |
| `/model-info` | GET | Model statistics |
| `/predict` | POST | Make prediction |
| `/upload-training-data` | POST | Upload training images |
| `/retrain` | POST | Trigger retraining |
| `/retraining-status` | GET | Check retraining progress |
| `/docs` | GET | API documentation |

---

## Performance Benchmarks

**Single Container:**
- Response Time: 200-400ms
- Throughput: ~30 req/s
- Max Concurrent Users: ~200

**Three Containers (Load Balanced):**
- Response Time: 150-250ms
- Throughput: ~80-100 req/s
- Max Concurrent Users: ~600

---

## Need Help?

- 📖 **Full Documentation:** See README.md
- 🐛 **Issues:** Create a GitHub issue
- 💬 **Questions:** your.email@example.com

---

**Happy Coding! 🎉**
