def bbox_center(box):
  X1, Y1, X2, Y2 = box
  center_x = int((X1 + X2) / 2)
  center_y = int((Y1 + Y2) / 2)
  return center_x, center_y

def points_distance(p1, p2):
  return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def convert_keypoints_to_xy_tuples(points):
    return [(int(points[i]), int(points[i + 1])) for i in range(0, len(points), 2)]