# models/resnet_model.py
from tensorflow.keras import models, layers

def create_resnet_model(input_shape=(224, 224, 3), num_classes=2):
    def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
        shortcut = x
        if conv_shortcut:
            shortcut = layers.Conv2D(filters, 1, strides=stride)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        
        return x
    
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    
    x = residual_block(x, 64, conv_shortcut=True)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2, conv_shortcut=True)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2, conv_shortcut=True)
    x = residual_block(x, 256)
    
    x = residual_block(x, 512, stride=2, conv_shortcut=True)
    x = residual_block(x, 512)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model
