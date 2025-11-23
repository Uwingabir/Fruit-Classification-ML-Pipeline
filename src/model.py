import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from datetime import datetime


class FruitClassifier:
    """
    Main model class for fruit classification.
    Supports multiple architectures and training/retraining capabilities.
    """
    
    def __init__(self, img_height=224, img_width=224, num_classes=6, model_type='mobilenet'):
        """
        Initialize the classifier.
        
        Args:
            img_height (int): Input image height
            img_width (int): Input image width
            num_classes (int): Number of classes
            model_type (str): Type of model architecture ('mobilenet', 'resnet', 'efficientnet', 'custom')
        """
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        self.history = None
        self.class_names = ['freshapples', 'freshbanana', 'freshoranges', 
                            'rottenapples', 'rottenbanana', 'rottenoranges']
        
    def build_model(self):
        """
        Build the model architecture based on model_type.
        
        Returns:
            keras.Model: Compiled model
        """
        if self.model_type == 'mobilenet':
            self.model = self._build_mobilenet()
        elif self.model_type == 'resnet':
            self.model = self._build_resnet()
        elif self.model_type == 'efficientnet':
            self.model = self._build_efficientnet()
        elif self.model_type == 'custom':
            self.model = self._build_custom_cnn()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def _build_mobilenet(self):
        """Build MobileNetV2-based model."""
        base_model = MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def _build_resnet(self):
        """Build ResNet50-based model."""
        base_model = ResNet50(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def _build_efficientnet(self):
        """Build EfficientNetB0-based model."""
        base_model = EfficientNetB0(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def _build_custom_cnn(self):
        """Build custom CNN architecture."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def train(self, train_generator, validation_generator, epochs=50, 
              model_save_path='models/fruit_classifier.h5', class_weights=None):
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs (int): Number of training epochs
            model_save_path (str): Path to save the best model
            class_weights (dict): Class weights for imbalanced data
            
        Returns:
            History: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save training history
        history_path = model_save_path.replace('.h5', '_history.json')
        self._save_history(history_path)
        
        return self.history
    
    def retrain(self, train_generator, validation_generator, epochs=20,
                model_path='models/fruit_classifier.h5', class_weights=None):
        """
        Retrain an existing model with new data.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs (int): Number of retraining epochs
            model_path (str): Path to existing model
            class_weights (dict): Class weights for imbalanced data
            
        Returns:
            History: Training history
        """
        # Load existing model
        if self.model is None:
            self.load_model(model_path)
        
        # Create backup
        backup_path = model_path.replace('.h5', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
        self.model.save(backup_path)
        print(f"Backup saved to: {backup_path}")
        
        # Unfreeze some layers for fine-tuning
        if self.model_type in ['mobilenet', 'resnet', 'efficientnet']:
            self._unfreeze_layers(num_layers=20)
        
        # Use lower learning rate for retraining
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                model_path,
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
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        # Retrain
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Save retraining history
        history_path = model_path.replace('.h5', f'_retrain_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        self._save_history(history_path)
        
        return self.history
    
    def _unfreeze_layers(self, num_layers=20):
        """Unfreeze the last num_layers for fine-tuning."""
        if self.model is None:
            return
        
        # Unfreeze the base model layers
        for layer in self.model.layers[0].layers[-num_layers:]:
            layer.trainable = True
    
    def evaluate(self, test_generator):
        """
        Evaluate the model on test data.
        
        Args:
            test_generator: Test data generator
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        # Get predictions
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, 
                                            target_names=self.class_names,
                                            output_dict=True)
        
        # Model evaluation
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(test_generator)
        
        results = {
            'accuracy': float(accuracy),
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        return results
    
    def save_model(self, path):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path):
        """Load model from file."""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from: {path}")
    
    def _save_history(self, path):
        """Save training history to JSON."""
        if self.history is None:
            return
        
        history_dict = {key: [float(val) for val in values] 
                       for key, values in self.history.history.items()}
        
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to: {path}")
    
    def get_model_summary(self):
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built"
        
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        return '\n'.join(summary)
