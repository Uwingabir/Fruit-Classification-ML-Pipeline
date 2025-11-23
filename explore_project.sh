#!/bin/bash

# Interactive Project Explorer
# Shows all outputs and components of the ML Pipeline

clear

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║       🔍 ML PIPELINE - PROJECT EXPLORER 🔍                    ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Function to pause
pause() {
    echo ""
    read -p "Press Enter to continue..."
    echo ""
}

# 1. Show Project Structure
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 PROJECT STRUCTURE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tree -L 2 -I 'venv|__pycache__|*.pyc|archive (2)' . 2>/dev/null || find . -maxdepth 2 -type f -o -type d | grep -v venv | grep -v __pycache__ | grep -v "archive (2)" | sort
pause

# 2. Show Documentation Files
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 DOCUMENTATION (What You Can Read)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. README.md - Complete project documentation"
echo "   Preview:"
head -20 README.md 2>/dev/null || echo "   File ready to view!"
echo "   ..."
echo ""
echo "2. QUICK_START.md - Step-by-step setup guide"
echo "3. PROJECT_SUMMARY.md - Executive summary"
echo "4. GITHUB_SETUP.md - GitHub upload instructions"
echo ""
echo "📖 To read any file:"
echo "   cat README.md          # View in terminal"
echo "   less README.md         # Scrollable view"
echo "   code README.md         # Open in VS Code"
echo "   xdg-open README.md     # Open in default editor"
pause

# 3. Show Python Scripts
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🐍 PYTHON SCRIPTS (What You Can Run)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Main Application:"
echo "  python app.py                    → Starts API server"
echo "  python project_info.py           → Shows project summary"
echo "  python test_api.py               → Tests API endpoints"
echo ""
echo "Modules (used by scripts):"
echo "  src/preprocessing.py             → Image preprocessing"
echo "  src/model.py                     → Model training"
echo "  src/prediction.py                → Predictions"
pause

# 4. Show Shell Scripts
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 SHELL SCRIPTS (What You Can Execute)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
ls -lh *.sh 2>/dev/null || echo "No shell scripts found"
echo ""
echo "Usage:"
echo "  ./setup.sh                       → Setup environment"
echo "  ./train_model.sh                 → Train the model"
echo "  ./deploy_aws.sh                  → Deploy to AWS"
echo "  ./explore_project.sh             → This script!"
pause

# 5. Show Jupyter Notebook
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📓 JUPYTER NOTEBOOK (Interactive Analysis)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ -f "notebook/fruit_classification.ipynb" ]; then
    echo "✓ Notebook found: notebook/fruit_classification.ipynb"
    echo ""
    echo "To open and see all outputs:"
    echo "  jupyter notebook notebook/fruit_classification.ipynb"
    echo ""
    echo "The notebook contains:"
    echo "  • Data exploration with charts"
    echo "  • Training process with progress"
    echo "  • Model evaluation with metrics"
    echo "  • Confusion matrix visualization"
    echo "  • Sample predictions with images"
else
    echo "✗ Notebook not found"
fi
pause

# 6. Show Web UI
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 WEB UI (Visual Interface)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ -f "static/index.html" ]; then
    echo "✓ Web UI found: static/index.html"
    echo ""
    echo "To see the UI:"
    echo ""
    echo "Step 1: Start the API server"
    echo "  python app.py"
    echo ""
    echo "Step 2: Open in browser"
    echo "  http://localhost:8000"
    echo ""
    echo "What you'll see:"
    echo "  • Model status dashboard"
    echo "  • Upload and predict images"
    echo "  • Interactive charts (Plotly)"
    echo "  • Real-time monitoring"
    echo "  • Retraining controls"
    echo ""
    echo "Or preview HTML directly:"
    echo "  xdg-open static/index.html"
    echo "  firefox static/index.html"
else
    echo "✗ Web UI not found"
fi
pause

# 7. Show Dataset Info
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 DATASET (Your Images)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ -d "archive (2)/dataset/train" ]; then
    echo "✓ Dataset found!"
    echo ""
    echo "To see your images:"
    echo "  nautilus 'archive (2)/dataset/train/'     # File manager"
    echo "  eog 'archive (2)/dataset/train/freshapples/*.png'  # Image viewer"
    echo ""
    echo "Image counts:"
    for dir in "archive (2)/dataset/train"/*; do
        if [ -d "$dir" ]; then
            count=$(find "$dir" -type f | wc -l)
            printf "  %-20s %5d images\n" "$(basename "$dir"):" "$count"
        fi
    done
else
    echo "⚠ Dataset not found in expected location"
fi
pause

# 8. Show Output Locations
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📤 WHERE TO FIND OUTPUTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "When you run training:"
echo "  models/fruit_classifier.h5              → Trained model"
echo "  models/fruit_classifier_history.json    → Training metrics"
echo ""
echo "When you run the API:"
echo "  Terminal output                         → Live logs"
echo "  http://localhost:8000/docs              → API documentation"
echo "  http://localhost:8000/metrics           → Prometheus metrics"
echo ""
echo "When you run load tests:"
echo "  Terminal output                         → Real-time stats"
echo "  results_stats.csv                       → Detailed results"
echo "  report.html                             → Visual report"
echo ""
echo "Docker logs:"
echo "  docker-compose logs                     → All container logs"
echo "  docker-compose logs -f app              → Follow app logs"
pause

# 9. Quick Demo Options
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎬 WHAT WOULD YOU LIKE TO SEE?"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Choose what to explore:"
echo ""
echo "1. View README.md                  (project documentation)"
echo "2. View QUICK_START.md             (setup guide)"
echo "3. View PROJECT_SUMMARY.md         (executive summary)"
echo "4. Show project statistics         (run project_info.py)"
echo "5. Open Jupyter notebook           (requires jupyter)"
echo "6. Preview Web UI in browser       (open HTML file)"
echo "7. List all available commands     (cheat sheet)"
echo "8. Exit"
echo ""
read -p "Enter your choice (1-8): " choice

case $choice in
    1)
        echo ""
        echo "Opening README.md..."
        less README.md 2>/dev/null || cat README.md
        ;;
    2)
        echo ""
        echo "Opening QUICK_START.md..."
        less QUICK_START.md 2>/dev/null || cat QUICK_START.md
        ;;
    3)
        echo ""
        echo "Opening PROJECT_SUMMARY.md..."
        less PROJECT_SUMMARY.md 2>/dev/null || cat PROJECT_SUMMARY.md
        ;;
    4)
        echo ""
        python3 project_info.py
        ;;
    5)
        echo ""
        echo "Starting Jupyter notebook..."
        jupyter notebook notebook/fruit_classification.ipynb 2>/dev/null || echo "Jupyter not installed. Run: pip install jupyter"
        ;;
    6)
        echo ""
        echo "Opening Web UI..."
        xdg-open static/index.html 2>/dev/null || firefox static/index.html 2>/dev/null || echo "Could not open browser. Open manually: static/index.html"
        ;;
    7)
        echo ""
        cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                     COMMAND CHEAT SHEET                       ║
╚═══════════════════════════════════════════════════════════════╝

📖 VIEW DOCUMENTATION
  cat README.md                        View main docs
  cat QUICK_START.md                   Quick setup guide
  cat PROJECT_SUMMARY.md               Project overview

🚀 RUN THE PROJECT
  ./setup.sh                           Setup environment
  ./train_model.sh                     Train model
  python app.py                        Start API server
  python project_info.py               Show project info
  python test_api.py                   Test API

📓 JUPYTER NOTEBOOK
  jupyter notebook                     Open notebook
  jupyter notebook notebook/fruit_classification.ipynb

🌐 WEB ACCESS (after starting API)
  http://localhost:8000               Main UI
  http://localhost:8000/docs          API documentation
  http://localhost:8000/health        Health check

🧪 LOAD TESTING
  locust -f locustfile.py --host=http://localhost:8000
  # Then open: http://localhost:8089

🐳 DOCKER
  docker-compose up -d                 Start all services
  docker-compose logs -f               View logs
  docker-compose ps                    Check status
  docker-compose down                  Stop all services

📊 VIEW RESULTS
  ls -lh models/                       See trained models
  cat models/*_history.json            Training metrics
  
🔍 EXPLORE FILES
  tree -L 2                            Project structure
  find . -name "*.py"                  Find Python files
  grep -r "TODO"                       Find todos

EOF
        ;;
    8)
        echo ""
        echo "Goodbye! 👋"
        exit 0
        ;;
    *)
        echo ""
        echo "Invalid choice. Please run the script again."
        ;;
esac

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ For more help, read QUICK_START.md or README.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
