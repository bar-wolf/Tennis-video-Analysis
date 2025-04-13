from video_analysis import read_video, show_frames, drow_frames_number

input_video_path = 'input_video.mp4'
frames = read_video(input_video_path)

# Court detection
from court_line_detector import CourtLineDetector
court_detector_model = 'keypoints_model.pth'
court_detector  = CourtLineDetector(court_detector_model)
court_detections = court_detector.detect_frame(frames)

from court_line_detector import CourtLineIntersections
court_intersections = CourtLineIntersections()
intersections = court_intersections.get_intersections(frames)
court_detections = court_intersections.update_keypoint_to_intersection(court_detections.reshape(-1, 2), intersections).reshape(-1)

# Player detection
from player_tracker import PlayerTracker
player_tracker = PlayerTracker()
player_detection = player_tracker.detect_video(frames)
player_detection = player_tracker.filter_playrs(court_detections, player_detection)

# Ball detection
from ball_tracker import BallTracker
ball_tracker_model = 'ball_model.pt'
ball_tracker = BallTracker(ball_tracker_model)
ball_detections = ball_tracker.detect_video(frames)
ball_detections = ball_tracker.interpolate_detections(ball_detections)

# Draw detections
output = player_tracker.draw_video(frames, player_detection)
output = ball_tracker.draw_video(frames, ball_detections)
output = court_detector.draw_video(frames, court_detections)
output = drow_frames_number(frames)

show_frames(frames)