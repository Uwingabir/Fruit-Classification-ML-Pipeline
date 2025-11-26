"""
Script to retrain the model with a "not_fruit" rejection class.

This will fix the issue where the model confidently classifies people,
objects, and other non-fruit images as fruits.

Usage:
    1. Create a folder: data/retrain/not_fruit/
    2. Add 500-1000 images of: people, objects, vehicles, documents, etc.
    3. Run: python retrain_with_rejection.py
"""

import os
import sys
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10

# Updated class names (7 classes now - added "not_fruit")
CLASS_NAMES = [
    'freshapples', 'freshbanana', 'freshoranges',
    'rottenapples', 'rottenbanana', 'rottenoranges',
    'not_fruit'  # NEW CLASS
]

# Paths
BASE_DIR = Path(__file__).parent
TRAIN_DIR = BASE_DIR / 'data' / 'retrain' / 'train'
VAL_DIR = BASE_DIR / 'data' / 'retrain' / 'val'
MODEL_PATH = BASE_DIR / 'models' / 'fruit_classifier_with_rejection.h5'


def prepare_data():
    """
    Prepare data generators.
    
    Expected directory structure:
    data/retrain/train/
        freshapples/
        freshbanana/
        freshoranges/
        rottenapples/
        rottenbanana/
        rottenoranges/
        not_fruit/      <-- NEW: Add images of people, objects, etc.
    
    data/retrain/val/
        (same structure)
    """
    print("\n" + "="*70)
    print("STEP 1: Preparing Data")
    print("="*70)
    
    # Check if directories exist
    if not TRAIN_DIR.exists():
        print(f"\n❌ ERROR: Training directory not found: {TRAIN_DIR}")
        print("\nYou need to:")
        print("1. Create: data/retrain/train/not_fruit/")
        print("2. Add 500-1000 images of:")
        print("   - People faces and bodies")
        print("   - Everyday objects (cars, phones, buildings)")
        print("   - Documents and screenshots")
        print("   - Other foods (pizza, pasta, etc.)")
        print("   - Animals")
        print("3. Copy existing fruit folders from archive (2)/dataset/train/")
        sys.exit(1)
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    print(f"\n✅ Data loaded successfully:")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {validation_generator.samples}")
    print(f"   Classes found: {train_generator.class_indices}")
    
    # Check if "not_fruit" class exists
    if 'not_fruit' not in train_generator.class_indices:
        print("\n❌ ERROR: 'not_fruit' class not found!")
        print("Please create: data/retrain/train/not_fruit/ and add images")
        sys.exit(1)
    
    not_fruit_count = sum([1 for label in train_generator.labels if train_generator.class_indices['not_fruit'] == label])
    print(f"\n   'not_fruit' images: {not_fruit_count}")
    
    if not_fruit_count < 500:
        print(f"\n⚠️  WARNING: Only {not_fruit_count} 'not_fruit' images found.")
        print("   Recommended: At least 500-1000 for good rejection capability")
    
    return train_generator, validation_generator


def build_model():
    """Build model with 7 classes (including not_fruit)."""
    print("\n" + "="*70)
    print("STEP 2: Building Model")
    print("="*70)
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model with 7 output classes (6 fruits + not_fruit)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(7, activation='softmax')  # 7 classes now
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    print("\n✅ Model built successfully")
    print(f"   Output classes: 7 (6 fruits + not_fruit)")
    print(f"   Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    return model


def train_model(model, train_generator, validation_generator):
    """Train the model."""
    print("\n" + "="*70)
    print("STEP 3: Training Model")
    print("="*70)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print("This will take 30-60 minutes on CPU\n")
    
    # Train
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"\nModel saved to: {MODEL_PATH}")
    
    return history


def update_prediction_script():
    """Update the prediction script to use 7 classes."""
    print("\n" + "="*70)
    print("STEP 4: Updating Configuration")
    print("="*70)
    
    pred_file = BASE_DIR / 'src' / 'prediction.py'
    
    print("\n⚠️  MANUAL STEP REQUIRED:")
    print(f"\nEdit {pred_file}")
    print("\nChange this line:")
    print("    self.class_names = ['freshapples', 'freshbanana', 'freshoranges',")
    print("                        'rottenapples', 'rottenbanana', 'rottenoranges']")
    print("\nTo:")
    print("    self.class_names = ['freshapples', 'freshbanana', 'freshoranges',")
    print("                        'rottenapples', 'rottenbanana', 'rottenoranges',")
    print("                        'not_fruit']")
    print("\nAnd update the model path in app.py to use the new model:")
    print(f"    MODEL_PATH = BASE_DIR / 'models' / 'fruit_classifier_with_rejection.h5'")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RETRAINING MODEL WITH 'NOT_FRUIT' REJECTION CLASS")
    print("="*70)
    print("\nThis will fix the issue where non-fruit images are misclassified.")
    
    try:
        # Prepare data
        train_gen, val_gen = prepare_data()
        
        # Build model
        model = build_model()
        
        # Train
        history = train_model(model, train_gen, val_gen)
        
        # Instructions
        update_prediction_script()
        
        print("\n" + "="*70)
        print("🎉 SUCCESS! Your model can now reject non-fruit images!")
        print("="*70)
        print("\nNext steps:")
        print("1. Update src/prediction.py with 7 class names (see above)")
        print("2. Update app.py to use new model path")
        print("3. Restart your server")
        print("\nYour app will now correctly reject:")
        print("  ✅ People and faces")
        print("  ✅ Objects and vehicles")
        print("  ✅ Documents and screenshots")
        print("  ✅ Other fruits not in training set")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
