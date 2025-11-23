"""
Project Information Display
Shows a summary of the ML Pipeline project
"""

import os
from pathlib import Path

def display_banner():
    """Display project banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║       🍎 FRUIT CLASSIFICATION ML PIPELINE 🍎                  ║
║                                                               ║
║       Complete End-to-End Machine Learning Project           ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def count_files(directory, extension):
    """Count files with specific extension"""
    try:
        return len(list(Path(directory).rglob(f"*.{extension}")))
    except:
        return 0

def get_file_size(filepath):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except:
        return 0

def display_project_stats():
    """Display project statistics"""
    print("\n📊 PROJECT STATISTICS")
    print("=" * 70)
    
    # Count code files
    py_files = count_files(".", "py")
    md_files = count_files(".", "md")
    html_files = count_files(".", "html")
    sh_files = count_files(".", "sh")
    
    print(f"  Python Files:      {py_files}")
    print(f"  Documentation:     {md_files}")
    print(f"  Shell Scripts:     {sh_files}")
    print(f"  HTML Files:        {html_files}")
    
    # Check for important files
    print("\n📁 PROJECT FILES")
    print("=" * 70)
    
    files_to_check = {
        "README.md": "Main documentation",
        "QUICK_START.md": "Quick start guide",
        "PROJECT_SUMMARY.md": "Project summary",
        "requirements.txt": "Python dependencies",
        "Dockerfile": "Docker configuration",
        "docker-compose.yml": "Multi-container setup",
        "app.py": "FastAPI application",
        "locustfile.py": "Load testing",
        "src/preprocessing.py": "Data preprocessing",
        "src/model.py": "Model architecture",
        "src/prediction.py": "Prediction logic",
        "static/index.html": "Web UI",
        "notebook/fruit_classification.ipynb": "Jupyter notebook"
    }
    
    for file, description in files_to_check.items():
        exists = "✓" if os.path.exists(file) else "✗"
        print(f"  {exists} {file:<35} {description}")

def display_dataset_info():
    """Display dataset information"""
    print("\n📦 DATASET INFORMATION")
    print("=" * 70)
    
    dataset_path = Path("archive (2)/dataset")
    
    if dataset_path.exists():
        train_path = dataset_path / "train"
        test_path = dataset_path / "test"
        
        classes = ["freshapples", "freshbanana", "freshoranges", 
                   "rottenapples", "rottenbanana", "rottenoranges"]
        
        print(f"  Dataset Path: {dataset_path}")
        print(f"\n  Classes:")
        
        total_train = 0
        total_test = 0
        
        for cls in classes:
            train_count = len(list((train_path / cls).glob("*"))) if (train_path / cls).exists() else 0
            test_count = len(list((test_path / cls).glob("*"))) if (test_path / cls).exists() else 0
            total_train += train_count
            total_test += test_count
            print(f"    {cls:<20} Train: {train_count:>5}  Test: {test_count:>5}")
        
        print(f"\n  Total Training Images: {total_train}")
        print(f"  Total Test Images:     {total_test}")
        print(f"  Total Images:          {total_train + total_test}")
    else:
        print("  ⚠ Dataset not found!")
        print("  Please ensure dataset is in 'archive (2)/dataset/' directory")

def display_model_info():
    """Display model information"""
    print("\n🤖 MODEL INFORMATION")
    print("=" * 70)
    
    model_path = Path("models/fruit_classifier.h5")
    
    if model_path.exists():
        size_mb = get_file_size(model_path)
        print(f"  Model File: {model_path}")
        print(f"  Model Size: {size_mb:.2f} MB")
        print(f"  Status: ✓ Ready")
    else:
        print("  Status: ✗ Model not trained yet")
        print("  Run: jupyter notebook notebook/fruit_classification.ipynb")
        print("  Or:  ./train_model.sh")

def display_api_endpoints():
    """Display API endpoints"""
    print("\n🌐 API ENDPOINTS")
    print("=" * 70)
    
    endpoints = [
        ("GET", "/", "API homepage"),
        ("GET", "/health", "Health check"),
        ("GET", "/model-info", "Model information"),
        ("POST", "/predict", "Make prediction"),
        ("POST", "/upload-training-data", "Upload training data"),
        ("POST", "/retrain", "Trigger retraining"),
        ("GET", "/retraining-status", "Retraining status"),
        ("GET", "/metrics", "Prometheus metrics"),
        ("GET", "/docs", "API documentation"),
    ]
    
    for method, endpoint, description in endpoints:
        print(f"  {method:<6} {endpoint:<30} {description}")

def display_commands():
    """Display useful commands"""
    print("\n🚀 QUICK COMMANDS")
    print("=" * 70)
    
    commands = [
        ("Setup Environment", "./setup.sh"),
        ("Train Model", "./train_model.sh"),
        ("Run API", "python app.py"),
        ("Run with Docker", "docker-compose up -d"),
        ("Run Load Test", "locust -f locustfile.py --host=http://localhost:8000"),
        ("Run Tests", "python test_api.py"),
        ("View Logs", "docker-compose logs -f"),
        ("Stop Docker", "docker-compose down"),
    ]
    
    for description, command in commands:
        print(f"  {description:<25} {command}")

def display_links():
    """Display important links"""
    print("\n🔗 IMPORTANT LINKS")
    print("=" * 70)
    
    links = [
        ("API Server", "http://localhost:8000"),
        ("API Docs", "http://localhost:8000/docs"),
        ("Web UI", "http://localhost:8000"),
        ("Locust UI", "http://localhost:8089"),
        ("Grafana", "http://localhost:3000 (admin/admin)"),
        ("Prometheus", "http://localhost:9090"),
        ("Load Balanced", "http://localhost (via nginx)"),
    ]
    
    for name, url in links:
        print(f"  {name:<20} {url}")

def main():
    """Main function"""
    display_banner()
    display_project_stats()
    display_dataset_info()
    display_model_info()
    display_api_endpoints()
    display_commands()
    display_links()
    
    print("\n" + "=" * 70)
    print("📚 For more information, see:")
    print("  • README.md - Complete documentation")
    print("  • QUICK_START.md - 10-minute setup guide")
    print("  • PROJECT_SUMMARY.md - Project overview")
    print("=" * 70)
    print("\n✨ Happy ML Engineering! ✨\n")

if __name__ == "__main__":
    main()
