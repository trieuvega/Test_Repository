import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Đường dẫn tới thư mục chứa dữ liệu
data_dir = 'data'

# Danh sách tên các lớp
class_names = []

for folderNames in os.listdir(data_dir):
    class_names.append(folderNames)
    # classes_dir.append(os.path.join(data_dir, folderNames))

# Thiết lập các thông số huấn luyện
input_shape = (64, 64, 3)
batch_size = 32
epochs = 10

# Đọc và chuẩn bị dữ liệu huấn luyện
X_train = []
y_train = []

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for filename in os.listdir(class_dir):
        image_path = os.path.join(class_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, input_shape[:2])
        X_train.append(image)
        y_train.append(class_names.index(class_name))

X_train = np.array(X_train)
y_train = np.array(y_train)

# Chuẩn hóa dữ liệu hình ảnh về khoảng từ 0 đến 1
X_train = X_train / 255.0

# Xây dựng mô hình CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

# Lưu mô hình đã huấn luyện
model.save('NhanDienChoMeo.h5')