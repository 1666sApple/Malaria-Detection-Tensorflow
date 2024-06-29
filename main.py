# main.py

import os
import tensorflow as tf
from data.dataloader import load_data, create_datasets
from models.resnet_model import create_resnet_model
from utils.image_processing import plot_images
from utils.train_utils import compile_and_train, plot_training_history

def main():
    # Set seeds for reproducibility
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    
    # Check for GPU availability
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("GPU is available. Using GPU.")
    else:
        print("No GPU available. Using CPU.")
    
    # Path of the dataset
    data_dir = "dataset/"
    categories = ['Parasitized', 'Uninfected']
    
    # Load data
    file_paths, labels = load_data(data_dir, categories)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(file_paths, labels)
    
    # Plot some sample images from the training dataset
    plot_images(train_dataset, categories)
    
    # Create and compile the model
    with tf.device('/GPU:0'):
        model = create_resnet_model(input_shape=(224, 224, 3))
    
    # Train the model
    with tf.device('/GPU:0'):
        history = compile_and_train(model, train_dataset, val_dataset)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    val_loss, val_acc = model.evaluate(val_dataset)
    print(f'Validation accuracy: {val_acc:.4f}')
    
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'Test accuracy: {test_acc:.4f}')
    
    # Save the model
    model.save('saved_model/parasitized_cell_classifier.h5')
    print("Model saved successfully.")

    # Load and verify the model
    loaded_model = tf.keras.models.load_model('saved_model/parasitized_cell_classifier.h5')
    print("Model loaded successfully.")
    test_loss, test_acc = loaded_model.evaluate(test_dataset)
    print(f'Loaded model - Test accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    main()
