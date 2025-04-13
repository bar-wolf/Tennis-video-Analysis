# Script extracted from section: Court Line Detector

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from scipy.spatial import cKDTree

class CourtLineDetector():
  def __init__(self, model_path, map_to='cpu'):
    self.model = models.resnet50(pretrained=True)
    self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
    self.model.load_state_dict(torch.load(model_path, map_location=map_to))

    self.transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  def detect_video(self, frames):
    court_line_detections = []
    for frame in frames:
      court_line_detections.append(self.detect_frame(frame))
    return court_line_detections

  def detect_frame(self, frame):
    if isinstance(frame, list):
      frame = frame[0]
    img = self.transforms(frame).unsqueeze(0)
    with torch.no_grad():
      output = self.model(img)
    keypoints = output.squeeze().cpu().numpy()
    # returns to original location
    orig_h, orig_w = frame.shape[:2]
    keypoints [0::2] *= orig_w / 224.0
    keypoints [1::2] *= orig_h / 224.0
    return keypoints

  def draw_video(self, video_frames, court_line_detections, color=(0, 0, 0)):
    output_frames = []
    if not isinstance(court_line_detections, list):
      for frame in video_frames:
        output_frames.append(self.draw_frame(frame, court_line_detections))
      return output_frames
    else:
      for frame, keypoints in zip(video_frames, court_line_detections):
        output_frames.append(self.draw_frame(frame, keypoints, color))
      return output_frames

  def draw_frame(self, frame, keypoints, color=(0, 0, 0)):
    for i in range(0, len(keypoints), 2):
      x, y = int(keypoints[i]), int(keypoints[i+1])
      cv2.putText(frame, str(i//2), (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
      cv2.circle(frame, (x, y), 5, color, -1)
    return frame


class CourtLineIntersections():
	def __init__(self, binary_threshold=200, line_threshold=250, min_line_length=100, max_line_gap=5, intersections_min_dist=10):
		self.binary_threshold = binary_threshold
		self.line_threshold = line_threshold
		self.min_line_length = min_line_length
		self.max_line_gap = max_line_gap
		self.intersections_min_dist = intersections_min_dist

	def get_intersections(self, image):
		if isinstance(image, list):
			image = image[0]
		binary = self._binary_image(image)
		lines = self._find_lines(binary)
		intersections = self._find_intersections(lines)
		return self._remove_neighbors_intersections(intersections)

	def update_keypoint_to_intersection(self, keypoints, intersections, min_distances=100):
		updated_keypoints = keypoints.copy()
		tree = cKDTree(intersections)
		distances, indices = tree.query(keypoints)
		for i, (dist, idx) in enumerate(zip(distances, indices)):
			if dist < min_distances:
				updated_keypoints[i] = intersections[idx]
		return updated_keypoints

	def draw_intersections(self, image, intersections):
		for x, y in intersections:
			cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

	def _binary_image(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY)[1]

	def _find_lines(self, binary_image):
		lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, self.line_threshold,
								minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
		return np.squeeze(lines) if lines is not None else np.array([])

	def _find_intersections(self, lines):
		intersections = []
		for i in range(len(lines)):
			for j in range(i + 1, len(lines)):
				x1, y1, x2, y2 = lines[i]
				x3, y3, x4, y4 = lines[j]
				A1, B1, C1 = self._line_equation(x1, y1, x2, y2)
				A2, B2, C2 = self._line_equation(x3, y3, x4, y4)

				det = A1 * B2 - A2 * B1
				if det != 0:  # lines are not parallel
					x = (C1 * B2 - C2 * B1) / det
					y = (A1 * C2 - A2 * C1) / det

					if self._is_on_line(x, y, x1, y1, x2, y2) or self._is_on_line(x, y, x3, y3, x4, y4):
						intersections.append((int(x), int(y)))
		return np.array(intersections)

	def _line_equation(self, x1, y1, x2, y2):
		A = y2 - y1
		B = x1 - x2
		C = A * x1 + B * y1
		return A, B, C

	def _is_on_line(self, px, py, x1, y1, x2, y2):
		return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)

	def _remove_neighbors_intersections(self, intersections):
		filtered = []
		for i, (x1, y1) in enumerate(intersections):
			keep = True
			for (x2, y2) in filtered:
				if np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < self.intersections_min_dist:
					keep = False  # stop checking if a close one is found
					break
			if keep:
				filtered.append((x1, y1))
		return np.array(filtered)
