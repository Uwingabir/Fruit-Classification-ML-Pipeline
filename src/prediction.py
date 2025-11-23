import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from preprocessing import ImagePreprocessor
import json
from datetime import datetime


class FruitPredictor:
    """
    Handles prediction operations for the fruit classifier.
    """
    
    def __init__(self, model_path='models/fruit_classifier.h5', img_height=224, img_width=224):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to the trained model
            img_height (int): Input image height
            img_width (int): Input image width
        """
        self.model_path = model_path
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.preprocessor = ImagePreprocessor(img_height, img_width)
        self.class_names = ['freshapples', 'freshbanana', 'freshoranges', 
                            'rottenapples', 'rottenbanana', 'rottenoranges']
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
    
    def predict_single_image(self, image_path):
        """
        Predict the class of a single image.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Preprocess image
            img = self.preprocessor.load_and_preprocess_image(image_path)
            img_batch = np.expand_dims(img, axis=0)
            
            # Predict
            predictions = self.model.predict(img_batch, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            # Create probability dictionary
            probabilities = {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def predict_from_bytes(self, image_bytes):
        """
        Predict from image bytes (for API usage).
        
        Args:
            image_bytes: Image bytes
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            # Preprocess image from bytes
            img = self.preprocessor.preprocess_from_bytes(image_bytes)
            img_batch = np.expand_dims(img, axis=0)
            
            # Predict
            predictions = self.model.predict(img_batch, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            # Create probability dictionary
            probabilities = {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
            
            # Interpret the prediction
            interpretation = self._interpret_prediction(predicted_class, confidence)
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'interpretation': interpretation,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple images.
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            list: List of prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single_image(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def _interpret_prediction(self, predicted_class, confidence):
        """
        Provide human-readable interpretation of the prediction.
        
        Args:
            predicted_class (str): Predicted class name
            confidence (float): Prediction confidence
            
        Returns:
            str: Interpretation
        """
        fruit_type = predicted_class.replace('fresh', '').replace('rotten', '')
        condition = 'fresh' if 'fresh' in predicted_class else 'rotten'
        
        confidence_level = 'very high' if confidence > 0.9 else \
                          'high' if confidence > 0.7 else \
                          'moderate' if confidence > 0.5 else 'low'
        
        interpretation = (
            f"This appears to be a {condition} {fruit_type} with "
            f"{confidence_level} confidence ({confidence:.2%}). "
        )
        
        if condition == 'rotten':
            interpretation += "The fruit shows signs of spoilage and should not be consumed."
        else:
            interpretation += "The fruit appears fresh and suitable for consumption."
        
        if confidence < 0.7:
            interpretation += " However, confidence is relatively low, so manual verification is recommended."
        
        return interpretation
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        try:
            # Get model file stats
            model_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
            model_modified = datetime.fromtimestamp(
                os.path.getmtime(self.model_path)
            ).isoformat()
            
            # Count parameters
            trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
            non_trainable_params = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
            
            info = {
                'model_path': self.model_path,
                'model_size_mb': round(model_size, 2),
                'last_modified': model_modified,
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'num_classes': len(self.class_names),
                'class_names': self.class_names,
                'trainable_parameters': int(trainable_params),
                'non_trainable_parameters': int(non_trainable_params),
                'total_parameters': int(trainable_params + non_trainable_params)
            }
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def validate_image(self, image_path):
        """
        Validate if an image file is valid and can be processed.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            dict: Validation result
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return {'valid': False, 'error': 'File does not exist'}
            
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in valid_extensions:
                return {'valid': False, 'error': f'Invalid file extension: {ext}'}
            
            # Try to load the image
            img = self.preprocessor.load_and_preprocess_image(image_path)
            
            return {
                'valid': True,
                'shape': img.shape,
                'size_kb': round(os.path.getsize(image_path) / 1024, 2)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def compare_predictions(self, image_paths):
        """
        Compare predictions across multiple images.
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            dict: Comparison statistics
        """
        predictions = self.predict_batch(image_paths)
        
        # Count predictions by class
        class_counts = {class_name: 0 for class_name in self.class_names}
        total_confidence = 0
        successful_predictions = 0
        
        for pred in predictions:
            if 'error' not in pred:
                class_counts[pred['predicted_class']] += 1
                total_confidence += pred['confidence']
                successful_predictions += 1
        
        avg_confidence = total_confidence / successful_predictions if successful_predictions > 0 else 0
        
        comparison = {
            'total_images': len(image_paths),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(image_paths) - successful_predictions,
            'average_confidence': round(avg_confidence, 4),
            'class_distribution': class_counts,
            'predictions': predictions
        }
        
        return comparison
