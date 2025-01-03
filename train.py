import os

from ultralytics import YOLO


config_path = './config.yaml'

# Load a model
model = YOLO("yolo11l.pt")

# Use the model
model.train(data=config_path, epochs=200, batch=32)  # train the model