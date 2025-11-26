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
            
            # IMPROVED: Check if this is likely a non-fruit image
            is_likely_fruit, warning_message = self._check_if_fruit(predictions[0], confidence)
            
            # Interpret the prediction
            interpretation = self._interpret_prediction(predicted_class, confidence)
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'interpretation': interpretation,
                'is_likely_fruit': is_likely_fruit,
                'warning': warning_message,
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
    
    def _check_if_fruit(self, prediction_probabilities, max_confidence):
        """
        Check if the image is likely a fruit based on prediction patterns.
        
        IMPORTANT: The model was ONLY trained on apples, bananas, and oranges.
        It WILL misclassify everything (people, objects, etc.) as one of these fruits.
        We use confidence patterns to detect this misclassification.
        
        Args:
            prediction_probabilities: Array of probabilities for all classes
            max_confidence: Maximum confidence score
            
        Returns:
            tuple: (is_likely_fruit: bool, warning_message: str)
        """
        # Calculate statistics about the prediction distribution
        std_dev = np.std(prediction_probabilities)
        max_prob = max_confidence
        sorted_probs = sorted(prediction_probabilities, reverse=True)
        second_max_prob = sorted_probs[1]
        third_max_prob = sorted_probs[2]
        prob_difference = max_prob - second_max_prob
        
        # Calculate entropy (uncertainty measure)
        probs_safe = prediction_probabilities + 1e-10
        entropy = -np.sum(probs_safe * np.log(probs_safe))
        
        # Decision criteria for non-fruit detection
        is_likely_fruit = True
        warning_message = None
        
        # CRITICAL: Even if confidence is high, if it's suspiciously high (>95%)
        # it might be overfitting to non-fruit patterns
        if max_confidence > 0.95 and prob_difference > 0.90:
            is_likely_fruit = False
            warning_message = (
                "🚨 SUSPICIOUS OVERCONFIDENCE (>95%) - The model is TOO confident! "
                "This often happens with NON-FRUIT images (people, objects, screenshots). "
                "The model was ONLY trained on apples, bananas, and oranges - "
                "it will incorrectly classify everything else as one of these fruits. "
                "⚠️ THIS IS LIKELY NOT A FRUIT!"
            )
        # Very low confidence (< 30%) - almost certainly not a fruit
        elif max_confidence < 0.30:
            is_likely_fruit = False
            warning_message = (
                "🚫 VERY LOW CONFIDENCE (<30%) - This is almost certainly NOT a fruit image! "
                "The model is extremely confused. Please upload an image of an apple, banana, or orange."
            )
        # Low confidence (30-50%) - probably not a fruit
        elif max_confidence < 0.50:
            is_likely_fruit = False
            warning_message = (
                "⚠️ LOW CONFIDENCE (30-50%) - This is probably NOT a fruit image! "
                "The model cannot recognize this as any fruit. "
                "Possible reasons: Wrong object type (person, screen, document, etc.), "
                "or unsupported fruit type."
            )
        # Medium-low confidence (50-70%) with small difference from second choice
        elif max_confidence < 0.70 and prob_difference < 0.20:
            is_likely_fruit = False
            warning_message = (
                "⚠️ UNCERTAIN (50-70%) - The model is confused between multiple classes. "
                "This might not be a clear fruit image, or the image quality is too poor. "
                "Try a clearer image of a single fruit."
            )
        # High confidence but all classes have similar low probabilities (flat distribution)
        elif max_confidence < 0.85 and std_dev < 0.15:
            is_likely_fruit = False
            warning_message = (
                "⚠️ FLAT DISTRIBUTION - All classes have similar low probabilities. "
                "The model doesn't recognize clear fruit patterns. "
                "This is likely NOT a fruit image."
            )
        # Medium confidence (70-85%) - could be fruit but uncertain
        elif max_confidence < 0.85:
            warning_message = (
                "⚡ MODERATE CONFIDENCE (70-85%) - Prediction might be correct but somewhat uncertain. "
                "For best results, use a clear, well-lit photo showing the fruit clearly."
            )
        
        return is_likely_fruit, warning_message
    
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
        
        # Warning for low confidence - likely not a fruit image
        if confidence < 0.5:
            interpretation = (
                f"⚠️ LOW CONFIDENCE WARNING ({confidence:.2%})! "
                f"This image may NOT be a fruit. The model was trained only on apples, bananas, and oranges. "
                f"If this is not one of these fruits, the prediction is unreliable. "
                f"Best guess: {condition} {fruit_type}, but please verify with a proper fruit image."
            )
        elif confidence < 0.7:
            interpretation = (
                f"⚠️ MODERATE CONFIDENCE ({confidence:.2%}). "
                f"This might be a {condition} {fruit_type}, but the model is not very certain. "
                f"Make sure the image contains one of these fruits: apples, bananas, or oranges."
            )
        else:
            interpretation = (
                f"This appears to be a {condition} {fruit_type} with "
                f"{confidence_level} confidence ({confidence:.2%}). "
            )
        
        if condition == 'rotten' and confidence >= 0.7:
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
