import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# --- 1. Preprocessing & Data Augmentation ---
# เนื่องจากเราฝึกจาก Scratch การทำ Augmentation ช่วยให้ Model ไม่ Overfit
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalization
    rotation_range=20,         # หมุนภาพ
    width_shift_range=0.2,     # เลื่อนภาพซ้ายขวา
    height_shift_range=0.2,    # เลื่อนภาพบนล่าง
    horizontal_flip=True,      # พลิกภาพกระจก
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(150, 150),    # ปรับขนาดภาพให้เท่ากัน
    batch_size=16,
    class_mode='binary'        # คัดแยก 2 ชนิดใช้ binary
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(150, 150),
    batch_size=1,              # สำหรับ Test ทีละภาพ
    class_mode='binary',
    shuffle=False
)

# --- 2. CNN Model Design (From Scratch) ---
model = models.Sequential([
    # Convolution Layer 1: ดึง Feature เบื้องต้น (เส้น, ขอบ)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    
    # Convolution Layer 2: ดึง Feature ที่ซับซ้อนขึ้น (รูปทรง)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Convolution Layer 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Flatten & Dense Layer (Classifier)
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),       # ลด Overfitting
    layers.Dense(1, activation='sigmoid') # Sigmoid สำหรับ Binary Classification
])

# --- 3. Cost Function & Optimizer ---
model.compile(
    optimizer='adam',                   # ปรับ Learning Rate อัตโนมัติ
    loss='binary_crossentropy',         # Cost function สำหรับ 2 คลาส
    metrics=['accuracy']
)

# --- 4. Training ---
history = model.fit(
    train_generator,
    epochs=20,                          # จำนวนรอบการฝึก
    validation_data=test_generator
)

# --- 5. Result Display (CLI & Plot) ---
print("\n--- สรุปผลการทดสอบ (Test Results) ---")
loss, acc = model.evaluate(test_generator)
print(f"Accuracy: {acc*100:.2f}%")

# ทำนายภาพใน Test set ทีละภาพ
test_generator.reset()
preds = model.predict(test_generator)
class_indices = train_generator.class_indices
labels = {v: k for k, v in class_indices.items()}

for i, p in enumerate(preds):
    result = labels[int(p > 0.5)] # ถ้า > 0.5 เป็นคลาส 1, ถ้า < เป็นคลาส 0
    print(f"ภาพที่ {i+1}: ผลลัพธ์คือ -> {result}")