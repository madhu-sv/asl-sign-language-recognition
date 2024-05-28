# src/ui.py

import tkinter as tk
from tkinter import Label
import threading
import cv2
import numpy as np
import tensorflow as tf

class ASLApp:
    def __init__(self, master):
        self.master = master
        master.title("ASL Real-time Recognition")

        self.label = Label(master, text="Starting...")
        self.label.pack()

        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.start()

    def load_model(self, model_path='../models/asl_model.keras'):
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_frame(self, frame, img_size=(64, 64)):
        img = cv2.resize(frame, img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def video_loop(self):
        self.load_model()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = self.preprocess_frame(frame)
            predictions = self.model.predict(img)
            predicted_class = np.argmax(predictions[0])

            self.label.config(text=f'Class: {predicted_class}')

            cv2.imshow('ASL Real-time Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLApp(root)
    root.mainloop()
