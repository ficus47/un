#import tensorflow as tf
from tensorflow.keras import layers, models
import moviepy.editor as mp
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.05)
train_generator = train_datagen.flow_from_directory(
    'NULL',
    target_size=(150, 150),
    batch_size=256,
    class_mode='sparse',  # Utilisez 'sparse' pour des étiquettes numériques
    shuffle=True,
    subset='training',  # Utiliser uniquement un sous-ensemble des données d'entraînement
        # Spécifiez la fraction de données à utiliser pour l'entraînement
)



def process_video(video_path):
    video = mp.VideoFileClip(video_path)
    duration = int(video.duration)

    for t in range(duration):
        frame = video.get_frame(t)
        # Save frame as .png without using cv2
        frame.save_frame(f"frame_{t}.png")


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

#model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
#model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))



model.compile(optimizer="adam", loss="mse", metrics=["mae", "accuracy"])

model.fit(train_generator, epochs=10, batch_size=256)



img_path = 'gaby.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

print(model.predict(img_array))
