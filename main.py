import tensorflow as tf
from tensorflow.keras import layers, models
import moviepy.editor as mp
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.utils import to_categorical



def get_image(dossier="test"):
  a = []
  for i in os.listdir(dossier):
    img = image.load_img(dossier + "/" + i, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    a.append(img_array)
  return a


train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'true',
    target_size=(150, 150),
    batch_size=512,
    class_mode='categorical',  # Utilisez 'sparse' pour des étiquettes numériques
    shuffle=True,
    subset='training',  # Utiliser uniquement un sous-ensemble des données d'entraînement
        # Spécifiez la fraction de données à utiliser pour l'entraînement
)

test_gen = ImageDataGenerator(rescale=1./255)
test = test_gen.flow_from_directory(
                                        "machin",
                                        target_size=(150, 150),
                                        batch_size=512,)



def process_video(video_path):
    video = mp.VideoFileClip(video_path)
    duration = int(video.duration)

    for t in range(duration):
        frame = video.get_frame(t)
        # Save frame as .png without using cv2
        frame.save_frame(f"frame_{t}.png")


model = models.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='elu'))
#model.add(layers.Dropout(0.11))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(128, activation='elu'))
#model.add(layers.Dropout(0.11))
model.add(layers.Dense(2, activation="softmax"))


e = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=20, verbose=2, restore_best_weights=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss="binary_crossentropy", metrics=["accuracy"])

model.fit(train_generator, epochs=250, callbacks=[e], verbose=2)

print(model.evaluate(test, verbose=2))
model.save("modelAI.keras")

while True:
  try:
    img_path = input("vous : ")
    if img_path == "exit":
       break
    
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    x = ["old", "young"]
    b = model.predict(img_array, verbose=2)
    print(x[b.argmax()], b)
  except Exception:
    pass

x = ["old", "young"]
a = get_image("test/young")
z = 0
images = []
rewards = []

for i in range(0):
  z += 1
  b = model.predict(i)

  print(x[b.argmax()], b)

  s = input("reward : ")
  reward =  int(s)if s.isdigit()else 0


  images.append(i)
  rewards.append(reward)

  if z % 8 == 0:
    # Convertir les listes en tableaux numpy
    images_array = np.concatenate(images, axis=0)
    rewards_array = np.array(rewards)
    rewards_categorical = to_categorical(rewards_array, num_classes=2)
    
        
    # Entraîner le modèle avec les données
    model.fit(images_array, rewards_array, batch_size=8, verbose=2)

    # Réinitialiser les listes pour le prochain lot
    images = []
    rewards = []



x = ["old", "young"]
a = get_image("test/old")
z = 0
for i in range(0):
  z += 1
  b = model.predict(i)
  
  print(x[b.argmax()], b)

  s = input("reward : ")
  reward =  int(s)if s.isdigit()else 0
  model.fit(i, [[reward]], verbose=2)

  if z // 8 == 0:
    model.evaluate(train_generator)
