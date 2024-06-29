# evaluate.py

import tensorflow as tf
from data.data_loader import load_data, create_datasets
from utils.image_processing import plot_images

def evaluate():
    # Path of the dataset
    data_dir = "dataset/"
    categories = ['Parasitized', 'Uninfected']
    
    # Load data
    file_paths, labels = load_data(data_dir, categories)
    
    # Create datasets
    _, val_dataset, test_dataset = create_datasets(file_paths, labels)
    
    # Load the model
    loaded_model = tf.keras.models.load_model('saved_model/parasitized_cell_classifier.h5')
    print("Model loaded successfully.")
    
    # Evaluate the model
    val_loss, val_acc = loaded_model.evaluate(val_dataset)
    print(f'Validation accuracy: {val_acc:.4f}')
    
    test_loss, test_acc = loaded_model.evaluate(test_dataset)
    print(f'Test accuracy: {test_acc:.4f}')

if __name__ == "__main__":
    evaluate()
