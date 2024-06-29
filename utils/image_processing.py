# utils/image_processing.py
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_images(dataset, categories, num_images=20):
    plt.figure(figsize=(15, 15))
    for i, (image, label) in enumerate(dataset.unbatch().take(num_images)):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(image.numpy())
        plt.title(f"Label: {categories[label.numpy()]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
