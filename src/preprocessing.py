import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

class ImagePreprocessor:
    """
    Handles all image preprocessing operations for the fruit classification model.
    """
    
    def __init__(self, img_height=224, img_width=224):
        """
        Initialize the preprocessor with target image dimensions.
        
        Args:
            img_height (int): Target height for images
            img_width (int): Target width for images
        """
        self.img_height = img_height
        self.img_width = img_width
        self.class_names = ['freshapples', 'freshbanana', 'freshoranges', 
                            'rottenapples', 'rottenbanana', 'rottenoranges']
    
    def load_and_preprocess_image(self, image_path, normalize=True):
        """
        Load and preprocess a single image.
        
        Args:
            image_path (str): Path to the image file
            normalize (bool): Whether to normalize pixel values to [0, 1]
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # Normalize if requested
        if normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
    
    def preprocess_from_bytes(self, image_bytes, normalize=True):
        """
        Preprocess image from bytes (useful for API uploads).
        
        Args:
            image_bytes: Image bytes
            normalize (bool): Whether to normalize pixel values
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert bytes to image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image from bytes")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # Normalize if requested
        if normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
    
    def create_data_generators(self, train_dir, test_dir, batch_size=32, augmentation=True):
        """
        Create data generators for training and validation.
        
        Args:
            train_dir (str): Path to training data directory
            test_dir (str): Path to test data directory
            batch_size (int): Batch size for training
            augmentation (bool): Whether to apply data augmentation
            
        Returns:
            tuple: (train_generator, test_generator)
        """
        if augmentation:
            # Data augmentation for training
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
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
        
        # Only rescaling for test data
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator, test_generator
    
    def batch_preprocess_images(self, image_paths, normalize=True):
        """
        Preprocess multiple images.
        
        Args:
            image_paths (list): List of image paths
            normalize (bool): Whether to normalize pixel values
            
        Returns:
            numpy.ndarray: Array of preprocessed images
        """
        images = []
        for path in image_paths:
            try:
                img = self.load_and_preprocess_image(path, normalize)
                images.append(img)
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                continue
        
        return np.array(images)
    
    def augment_image(self, image):
        """
        Apply random augmentation to an image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Augmented image
        """
        # Random rotation
        if np.random.random() > 0.5:
            angle = np.random.randint(-20, 20)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        # Random flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness
        if np.random.random() > 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv = hsv.astype(np.float32)
            hsv[:, :, 2] *= np.random.uniform(0.7, 1.3)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            hsv = hsv.astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image
    
    def get_class_weights(self, train_dir):
        """
        Calculate class weights for imbalanced datasets.
        
        Args:
            train_dir (str): Path to training directory
            
        Returns:
            dict: Class weights
        """
        class_counts = {}
        for class_name in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_name)
            if os.path.isdir(class_path):
                count = len(os.listdir(class_path))
                class_counts[class_name] = count
        
        total = sum(class_counts.values())
        class_weights = {i: total / (len(class_counts) * count) 
                        for i, (_, count) in enumerate(class_counts.items())}
        
        return class_weights
