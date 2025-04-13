from ultralytics import YOLO
import cv2
import pandas as pd

class BallTracker():
  def __init__(self, model_path):
    self.model = YOLO(model_path)

  def detect_video(self, frames):
    ball_detections = []
    for frame in frames:
      ball_detections.append(self.detect_frame(frame))
    return ball_detections

  def detect_frame(self, frame):
    results = self.model.predict(frame, conf=0.1)[0]
    ball = {}
    for box in results.boxes:
      bbox = box.xyxy.tolist()[0]
      ball[1] = bbox
    return ball

  def interpolate_detections(self, ball_detections):
    ball_detections = [x.get(1, []) for x in ball_detections]

    # convert to DataFrame and interpolate
    df_ball_detections = pd.DataFrame(ball_detections, columns=['x1', 'y1', 'x2', 'y2'])
    df_ball_detections = df_ball_detections.interpolate()
    df_ball_detections = df_ball_detections.bfill()  # copie the first detection to the first frames
    ball_detections = [{1:x} for x in df_ball_detections.to_numpy().tolist()]  # returns to the input format
    return ball_detections

  def draw_video(self, video_frames, ball_detections):
    output_frames = []
    for frame, plyer in zip(video_frames, ball_detections):
      output_frames.append(self.draw_frame(frame, plyer))
    return output_frames

  def draw_frame(self, frame, ball):
    for track_id, bbox in ball.items():
      x1, y1, x2, y2 = bbox
      # drow bboxs around the ball
      cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
    return frame