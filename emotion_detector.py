#!/usr/bin/env python3
"""
Emotion Detection System for Raspberry Pi
Detects emotions from facial expressions using OpenCV and a pre-trained model (simulated)
"""

import cv2
import numpy as np
from datetime import datetime
import os

class EmotionDetector:
    """
    Real-time emotion detection from webcam feed
    Supports 7 emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised
    """
    
    def __init__(self):
        # Emotion labels
        self.emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        
        # Initialize face detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # For this demo, we'll use a simple brightness-based emotion simulator
        # In production, you would load a trained CNN model here
        self.model_loaded = False
        
        print("✅ Emotion Detector initialized")
        print("Supported emotions:", self.emotions)
    
    def preprocess_face(self, face_img):
        """Preprocess face image for emotion detection"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 48, 48, 1)
        return reshaped
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image (simplified heuristic demo)"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray)
        variance = np.var(gray)
        
        if avg_intensity > 150:
            emotion_idx = 3  # Happy
        elif avg_intensity < 80:
            emotion_idx = 5  # Sad
        elif variance > 1000:
            emotion_idx = 6  # Surprised
        else:
            emotion_idx = 4  # Neutral
        
        confidences = np.zeros(7)
        confidences[emotion_idx] = 0.85
        
        return emotion_idx, confidences
    
    def draw_results(self, frame, x, y, w, h, emotion, confidence):
        """Draw bounding box and emotion label on frame"""
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{emotion}: {confidence:.2f}"
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame, 
            (x, y - text_height - 10), 
            (x + text_width, y), 
            (0, 255, 0), 
            -1
        )
        cv2.putText(
            frame, 
            label, 
            (x, y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 0, 0), 
            2
        )
    
    def run(self, camera_index=0):
        """Run real-time emotion detection"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("\n🎥 Emotion Detection Started")
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        print("-" * 40)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                emotion_idx, confidences = self.predict_emotion(face_roi)
                emotion = self.emotions[emotion_idx]
                confidence = confidences[emotion_idx]
                self.draw_results(frame, x, y, w, h, emotion, confidence)
                
                if frame_count % 30 == 0:
                    print(f"Detected: {emotion} (Confidence: {confidence:.2f})")
            
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Emotion Detector', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("🛑 Emotion Detection stopped")


def main():
    """Main function to run emotion detector"""
    print("=" * 50)
    print("🧠 Raspberry Pi Emotion Detection System")
    print("=" * 50)
    
    detector = EmotionDetector()
    try:
        detector.run(camera_index=0)
    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user")
    except Exception as e:
        print(f"\n⚠️ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()