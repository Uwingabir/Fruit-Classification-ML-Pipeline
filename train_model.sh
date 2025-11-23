#!/bin/bash

# Quick training script for the fruit classification model

echo "Starting model training..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Run training script
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.append('src')

from preprocessing import ImagePreprocessor
from model import FruitClassifier
import os

# Configuration
TRAIN_DIR = 'archive (2)/dataset/train'
TEST_DIR = 'archive (2)/dataset/test'
MODEL_PATH = 'models/fruit_classifier.h5'
MODEL_TYPE = 'mobilenet'  # Options: mobilenet, resnet, efficientnet, custom
EPOCHS = 50
BATCH_SIZE = 32

print("\n" + "="*60)
print("  Fruit Classification Model Training")
print("="*60)
print(f"\nConfiguration:")
print(f"  Model Type: {MODEL_TYPE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Training Data: {TRAIN_DIR}")
print(f"  Test Data: {TEST_DIR}")
print(f"  Model Save Path: {MODEL_PATH}")
print("="*60 + "\n")

# Check if dataset exists
if not os.path.exists(TRAIN_DIR):
    print(f"Error: Training directory not found: {TRAIN_DIR}")
    print("Please organize your dataset in the correct structure.")
    sys.exit(1)

# Initialize preprocessor
print("Initializing preprocessor...")
preprocessor = ImagePreprocessor(img_height=224, img_width=224)

# Create data generators
print("Creating data generators...")
train_gen, val_gen, test_gen = preprocessor.create_data_generators(
    TRAIN_DIR, 
    TEST_DIR, 
    batch_size=BATCH_SIZE,
    augmentation=True
)

print(f"  Training samples: {train_gen.samples}")
print(f"  Validation samples: {val_gen.samples}")
print(f"  Test samples: {test_gen.samples}")
print(f"  Number of classes: {train_gen.num_classes}")
print(f"  Class names: {list(train_gen.class_indices.keys())}")

# Calculate class weights
print("\nCalculating class weights...")
class_weights = preprocessor.get_class_weights(TRAIN_DIR)
print(f"  Class weights: {class_weights}")

# Initialize model
print(f"\nBuilding {MODEL_TYPE} model...")
classifier = FruitClassifier(
    img_height=224,
    img_width=224,
    num_classes=6,
    model_type=MODEL_TYPE
)
model = classifier.build_model()

# Print model summary
print("\nModel Architecture:")
print(classifier.get_model_summary())

# Train model
print(f"\nStarting training for {EPOCHS} epochs...")
print("This may take a while. Progress will be shown below.\n")

history = classifier.train(
    train_gen,
    val_gen,
    epochs=EPOCHS,
    model_save_path=MODEL_PATH,
    class_weights=class_weights
)

# Evaluate on test set
print("\nEvaluating model on test set...")
results = classifier.evaluate(test_gen)

print("\n" + "="*60)
print("  Training Complete!")
print("="*60)
print(f"\nTest Results:")
print(f"  Accuracy: {results['accuracy']:.4f}")
print(f"  Precision: {results['test_precision']:.4f}")
print(f"  Recall: {results['test_recall']:.4f}")
print(f"  Loss: {results['test_loss']:.4f}")
print(f"\nModel saved to: {MODEL_PATH}")
print("="*60 + "\n")

print("✓ Training completed successfully!")
print("\nNext steps:")
print("1. Review results in the saved model file")
print("2. Run the API: python app.py")
print("3. Test predictions via the web UI")

PYTHON_SCRIPT

echo ""
echo "✓ Script completed"
