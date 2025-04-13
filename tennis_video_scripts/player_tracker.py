# Script extracted from section: Player Tracker

from ultralytics import YOLO
import cv2
import numpy as np

class PlayerTracker():
  def __init__(self, model_path='yolov8n.pt'):
    self.model = YOLO(model_path)

  def detect_video(self, frames):
    player_detections = []
    for frame in frames:
      player_detections.append(self.detect_frame(frame))
    return player_detections

  def detect_frame(self, frame):
    results = self.model.track(frame, persist=True)[0]
    player = {}
    for box in results.boxes:
      track_id = int(box.id.tolist()[0])
      bbox = box.xyxy.tolist()[0]
      # keep only 'person' abjects
      if results.names[box.cls.tolist()[0]] == 'person':
        player[track_id] = bbox
    return player

  def draw_video(self, video_frames, plyer_detections, color=(255, 0, 0)):
    output_frames = []
    for frame, plyer in zip(video_frames, plyer_detections):
      output_frames.append(self.draw_frame(frame, plyer, color))
    return output_frames

  def draw_frame(self, frame, plyer, color=(255, 0, 0)):
    for track_id, bbox in plyer.items():
      x1, y1, x2, y2 = bbox
      # add 'Plyer ID'
      text = "Player: " + str(track_id)
      cv2.putText(frame, text, (int(x1), int(y1) -10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 2)
      # drow bboxs around player
      cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return frame

  def filter_playrs(self, keypoints, player_detections):
    true_playrs_detections = []
    for detection in player_detections:
      # calculate mean distance of each "player" to the keypoints
      players_distance = self._players_mean_distance_to_keypoints(keypoints, detection)
      # find the players (the two closest to the keypoints)
      players_distance.sort(key=lambda x: x[1])
      true_players_id = [players_distance[0][0], players_distance[1][0]]
      # extract the firs two closest from the player detections
      true_player = {}
      for track_id, bbox in detection.items():
        if track_id in true_players_id:
          true_player[track_id] = bbox
      true_playrs_detections.append(true_player)
      # maintain player id by their position on court
      true_playrs_detections = self._maintain_players_id_by_position(true_playrs_detections)
    return true_playrs_detections

  def _players_mean_distance_to_keypoints(self, keypoints, player_detections):
    players_distance = []
    for track_id, bbox in player_detections.items():
      mean_distance = self._calculate_mean_distance(bbox, keypoints)
      players_distance.append((track_id, float(mean_distance)))
    return players_distance

  def _calculate_mean_distance(self, player_bbox, keypoints):
      player_mean_distance = []
      # calculate player bbox center
      player_center = bbox_center(player_bbox)
      keypoints = convert_keypoints_to_xy_tuples(keypoints)
      for keypoint in keypoints:
        distance = points_distance(player_center, keypoint)
        player_mean_distance.append(distance)
      return np.mean(player_mean_distance)

  def _maintain_players_id_by_position(self, true_playrs_detections):
    consistent_id_detection = []
    for detection in true_playrs_detections:
      new_id_detection = {}
      bboxs = list(detection.values())

      y_position_1 = bboxs[0][1]
      y_position_2 = bboxs[1][1]

      if y_position_2 < y_position_1:
        new_id_detection[1] = bboxs[0]
        new_id_detection[2] = bboxs[1]
      else:
        new_id_detection[1] = bboxs[1]
        new_id_detection[2] = bboxs[0]

      consistent_id_detection.append(new_id_detection)
    return consistent_id_detection