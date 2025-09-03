import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os

os.makedirs("models", exist_ok=True)

# تحميل البيانات
X = np.load("extracted/data.npy", allow_pickle=True)
y = np.load("extracted/labels.npy", allow_pickle=True)

# التحقق من عدد العينات
n_samples = len(X)
if n_samples > 1:
    # إذا كان هناك أكثر من عينة واحدة، نقسم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=True
    )
else:
    # إذا كانت العينات قليلة جدًا، نأخذ كل البيانات للتدريب
    X_train, y_train = X, y
    X_test, y_test = None, None
    print("عدد العينات قليل جدًا، سيتم استخدام كل البيانات للتدريب فقط.")

# بناء نموذج LSTM
model = models.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(30, X.shape[2])),
    layers.LSTM(64),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(set(y)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# تدريب النموذج
if X_test is not None:
    model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
else:
    model.fit(X_train, y_train, epochs=30)

# حفظ النموذج
model.save("models/behavior_model.h5")
print("Model saved in models/behavior_model.h5")
