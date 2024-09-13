import streamlit as stl
import numpy as np
import os

data = 'data'
classes = []
classes_dir = []
image_shape = (64, 64, 3)

for class_names in os.listdir(data):
    classes.append(class_names)
    classes_dir.append(os.path.join(data, class_names))

stl.write(data)
stl.write(classes)
stl.write(image_shape)