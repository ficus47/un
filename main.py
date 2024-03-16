#import tensorflow as tf
from tensorflow.keras import layers, models
import moviepy.editor as mp

def process_video(video_path):
    video = mp.VideoFileClip(video_path)
    duration = int(video.duration)

    for t in range(duration):
        frame = video.get_frame(t)
        # Save frame as .png without using cv2
        frame.save_frame(f"frame_{t}.png")


#model qui predit l'age de quelqun a partir de la photo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150,)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
