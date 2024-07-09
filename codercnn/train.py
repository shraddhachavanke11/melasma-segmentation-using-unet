import os
import mrcnn.model as modellib
from custom_dataset import CustomDataset
from mrcnn_config import Config

# Initialize model
model = modellib.MaskRCNN(mode="training", config=Config(), model_dir="./logs")
model.keras_model.summary()

# Load dataset
dataset_train = CustomDataset()
dataset_train.load_custom("C:/Users/asus/Desktop/melasmafinal/shraddhamodifieddataset/final", "train")
dataset_train.prepare()

# Start training
model.train(dataset_train, dataset_train, learning_rate=Config.LEARNING_RATE, epochs=10, layers='heads')
