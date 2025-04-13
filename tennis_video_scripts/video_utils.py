import cv2
import matplotlib.pyplot as plt

def read_video(video_path):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error reading video file")
    return

  frames = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  cap.release()
  return frames

def save_video(output_frames, output_path, frame_per_sec = 30):
  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  out = cv2.VideoWriter(output_path, fourcc, frame_per_sec, (output_frames[0].shape[1], output_frames[0].shape[0]))
  for frame in output_frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  out.release()

def show_frames(frames_list, idx=(0, 5)):
  if not isinstance(frames_list, list):
      plt.imshow(frames_list)
      plt.show()
  elif isinstance(idx, tuple):
    for i in range(idx[0], idx[1]):
      plt.imshow(frames_list[i])
      plt.show()
  elif isinstance(idx, int):
    plt.imshow(frames_list[idx])
    plt.show()

def drow_frames_number(frame, color=(0, 255, 255)):
  for i, frame in enumerate(frame):
    text = "Frame: " + str(i+1)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)