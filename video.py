import moviepy.editor as mp
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os 
import random
import numpy as np
import imageio

def preprocess_image(file):
    img = image.load_img(file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def process_video(video_path):
    video = mp.VideoFileClip(video_path)
    duration = int(video.duration)

    for t in range(duration):
      if random.randint(0, 10) == 1:
        frame = video.get_frame(t)
        # Enregistrer chaque image sous forme de fichier PNG
        imageio.imwrite(f"output/frame_{t}.png", frame)

model = tf.keras.models.load_model("modelAI.keras")
process_video("video.mp4")

x = ["old", "young"]
dic = {"old" : -1, "young" : 1}
note = []

for i in os.listdir("output"):
  img = preprocess_image("output/" + i)
  a = model.predict(img).argmax()
  note.append(dic[x[a]])
        
note = sum(note)
if note >= 0:
    open("note.txt", "w").write("1")
else:
    open("note.txt", "w").write("0")

