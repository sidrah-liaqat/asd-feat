# Author: Sidrah Liaqat (sidrah.liaqat@uky.edu) University of Kentucky
from __future__ import print_function, division
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image, ImageDraw

w = 720
h = 480
ht = 30

def show_landmarks_pillow(sample, act_image):
    landmarks, eyelandmarks, headpose, objectpose, gazeangle = sample['landmarks'], sample['eyelandmarks'], sample[
        'headpose'], sample['objectpose'], sample['gazeangle']

    PIL_image = Image.fromarray(np.uint8(act_image))
    # PIL_image = Image.fromarray(255 * np.zeros((h, w, 3), np.uint8))
    """assuming img already RGB"""
    draw = ImageDraw.Draw(PIL_image, 'RGBA')

    draw.rectangle(((headpose[0, 0] * w, headpose[0, 1] * h),
                    (headpose[1, 0] * w, headpose[1, 1] * h)), width=4)

    draw.rectangle(((objectpose[0, 0] * w, objectpose[0, 1] * h),
                    (max(objectpose[1, 0], objectpose[0, 0]) * w, max(objectpose[1, 1], objectpose[0, 1]) * h)),
                   width=4)

    # for drawing head ellipse in top right corner of head box
    """
    tlx = (headpose[1, 0] + headpose[0, 0]) / 2
    bry = (headpose[0, 1] + headpose[1, 1]) / 2
    draw.ellipse((tlx * w, headpose[0, 1] * h,
                  headpose[1, 0] * w, bry * h), fill=(255, 0, 0, 125))
    """
    # for drawing head ellipse in full head bounding box
    draw.ellipse((headpose[0, 0] * w, headpose[0, 1] * h,
                  headpose[1, 0] * w, headpose[1, 1] * h), fill=(255, 0, 0, 125))
    xy = list()
    # for i in range(eyelandmarks.shape[0]):
    #    xy.append((eyelandmarks[i, 0] * w, eyelandmarks[i, 1] * h))
    # draw.point(xy, fill=None)

    xy = list()
    for i in range(landmarks.shape[0]):
        xy.append((landmarks[i, 0] * w, landmarks[i, 1] * h))
    draw.point(xy, fill=None)

    tpi = torch.Tensor([np.pi])
    gaze_xy1 = list()
    # starting point is eye landmark 36 and 42 for both eyes
    x1 = landmarks[36, 0] * w
    y1 = landmarks[36, 1] * h

    gaze_xy1.append((x1, y1))
    # x2 = x1 + 100 * torch.sin(tpi*gazeangle[0,0]+tpi/2)
    # y2 = y1 + 100 * torch.(tpi*gazeangle[0,1]+tpi/2)
    # x2 = x1 + 100*np.sin(gazeangle[0,0])
    # y2 = y1 + 100*np.sin(gazeangle[0,1])
    # gazeangle = gazeangle.squeeze(0)
    x2 = x1 + 100 * gazeangle[0, 0]
    y2 = y1 + 100 * gazeangle[0, 1]

    gaze_xy1.append((x2, y2))

    gaze_xy2 = list()
    # starting point is face landmark 27 and 55 for both eyes
    x3 = landmarks[42, 0] * w
    y3 = landmarks[42, 1] * h
    gaze_xy2.append((x3, y3))

    # x4 = x3 + 100*np.sin(gazeangle[0,0])
    # y4 = y3 + 100*np.sin(gazeangle[0,1])
    x4 = x3 + 100 * gazeangle[0, 0]
    y4 = y3 + 100 * gazeangle[0, 1]
    gaze_xy2.append((x4, y4))

    draw.text((50, 50), 'Look Face', (255, 255, 255))
    # draw.text((500, 60), str(gazeangle[1]), (255, 255, 255))

    # draw.line(gaze_xy1, width=2, fill=(255, 0, 0, 125))
    # draw.line(gaze_xy2, width=2, fill=(255, 0, 0, 125))

    """
    gaze_ray_head = list()
    gaze_ray_head.append((x1, y1))
    gaze_ray_head.append(((headpose[0, 0]+headpose[1, 0]) * w/2, (headpose[0, 1]+headpose[1, 1]) * h/2))
    gaze_ray_toy = list()
    gaze_ray_toy.append((x1, y1))
    gaze_ray_toy.append(((objectpose[0, 0]+objectpose[1, 0]) * w/2, (objectpose[0, 1]+objectpose[1, 1]) * h/2))
    head_pose = list()
    head_pose.append((headpose[0, 0] * w, headpose[0, 1] * h))
    head_pose.append((headpose[1, 0] * w, headpose[1, 1] * h))
    toy_pose = list()
    toy_pose.append((objectpose[0, 0] * w, objectpose[0, 0] * h))
    toy_pose.append((objectpose[1, 0] * w, objectpose[1, 0] * h))
    #if intersects_angle((x1,x2), gazeangle[0,0], gazeangle[0,1], 
    #    (headpose[0, 0] * w, headpose[0, 1] * h), 
    #    (headpose[1, 0] * w, headpose[1, 1] * h), 15):
    #    draw.line(gaze_xy1, width=2, fill=(255, 0, 0, 125))
    draw.line(gaze_ray_head, width=4, fill=(0, 255, 0, 125))
    draw.line(gaze_ray_toy, width=4, fill=(0, 0, 255, 125))
    """
    return PIL_image


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=4, marker='.', c='b')

    plt.pause(0.3)  # pause a bit so that plots are updated


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    # images_batch, lmk_x_batch, lmk_y_batch = \
    #        sample_batched['image'], sample_batched['lmk_x'], sample_batched['lmk_y']

    images_batch, landmarks_batch, headpose_batch = \
        sample_batched['image'], sample_batched['landmarks'], sample_batched['headpose']
    batch_size = len(images_batch)
    im_size = images_batch.size(3)
    w = images_batch.size(3)
    h = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)).astype(int))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() * w + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() * h + grid_border_size,
                    s=5, marker='.', c='b')

        plt.scatter(headpose_batch[i, :, 0].numpy() * w + i * im_size + (i + 1) * grid_border_size,
                    headpose_batch[i, :, 0].numpy() * h + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


def fill_plane(plane, mod):
    h, w = plane.shape

    try:

        m = np.max(mod) + 1  # Get the maximum value from mod array
    except:
        print("pause")
    if (max(mod) == 1):
        mod = mod * frame
    # Iterate over each index of mod
    for i, val in enumerate(mod):
        # Calculate the number of ones to fill in the corresponding column
        num_ones = min(int(np.floor(val * 15 / frame)), h)

        # Fill the column with ones from bottom to top
        plane[h - num_ones:, i] = 1

    return plane


def label_to_img(label, ht=30, color=[64, 128, 192, 125]):
    # this function gives out one row of color
    # converts 1d data to 3D image like tensor
    w = len(label)
    a = np.array(label).astype(np.uint8)
    b = np.array(np.ones((ht), np.uint8))
    # b = np.arange(ht)
    # Rescale values in b to match the range of values in label
    # b_rescaled = (b / (ht - 1)) * (np.max(a) - np.min(a)) + np.min(a)
    c = a[:, np.newaxis] * b
    c = np.transpose(c)
    # Create a 2D array where each row represents the height of the stripe
    # c = np.tile(b_rescaled[:, np.newaxis], (w, 1))

    # Example usage
    plane = np.zeros((15, w), dtype=int)

    c = fill_plane(plane, label)
    img = c[:, :, np.newaxis] * color
    return img.astype(np.uint8)


def pred_to_stats(array):
    # This function takes frame by frame ground truth or prediction
    # (clustered, ideally) and converts it to duration and frequency
    # The objective is to compare the DL model predictions with
    # groundtruth statistics (duration and frequency of events)

    # array: 1D input numpy array consisting of ones and zeros

    dur = np.sum(array) / 30.0
    freq = np.size(np.where(np.diff(array, prepend=0, append=0) == 1))
    return dur, freq


def make_fullimage(label1, label15, label2, label3):
    # this function takes a full length label and prediction
    # then forms image out of it
    # it is expected that label and prediction will be of same length
    fullimage = 255 * np.zeros((h + 120 + 120, w, 3), np.uint8)
    # label color blue
    # prediction color red
    blue = (0, 0, 200)
    red = (255, 0, 0)
    green = (0, 200, 0)
    orange = (255, 215, 0)

    ind, rem = divmod(len(label1), w)
    if rem == 0:
        print('pause')
    i = 0
    for i in range(ind):
        fullimage[i * 2 * ht + 0:i * 2 * ht + 15, :, :] = \
            label_to_img(np.array(label1[i * w:(i + 1) * w], dtype=int), ht=15, color=blue)
        fullimage[i * 2 * ht + 15:i * 2 * ht + 30, :, :] = \
            label_to_img(np.array(label15[i * w:(i + 1) * w], dtype=int), ht=15, color=red)  # red
        fullimage[i * 2 * ht + 30:i * 2 * ht + 45, :, :] = \
            label_to_img(np.array(label2[i * w:(i + 1) * w], dtype=int), ht=15, color=green)  # green
        fullimage[i * 2 * ht + 45:i * 2 * ht + 60, :, :] = \
            label_to_img(np.array(label3[i * w:(i + 1) * w], dtype=int), ht=15, color=orange)  # orange
    # drawing the remainder part of timeline
    if (rem > 0):
        i += 1

        fullimage[i * 2 * ht + 0:i * 2 * ht + 15, 0:rem, :] = \
            label_to_img(np.array(label1[i * w:i * w + rem], dtype=int), ht=15, color=blue)
        fullimage[i * 2 * ht + 15:i * 2 * ht + 30, 0:rem, :] = \
            label_to_img(np.array(label15[i * w:i * w + rem], dtype=int), ht=15, color=red)
        fullimage[i * 2 * ht + 30:i * 2 * ht + 45, 0:rem, :] = \
            label_to_img(np.array(label2[i * w:i * w + rem], dtype=int), ht=15, color=green)
        fullimage[i * 2 * ht + 45:i * 2 * ht + 60, 0:rem, :] = \
            label_to_img(np.array(label3[i * w:i * w + rem], dtype=int), ht=15, color=orange)

    return fullimage


def mark_curr_frame(image, fidx):
    f_ind, f_rem = divmod(fidx, w)
    PIL_image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(PIL_image)
    draw.rectangle(((int(f_rem - 1), int(f_ind * 60)),
                    (int(f_rem + 1), int((f_ind + 1) * 60))), width=4)
    # draw.rectangle(((10, 10),
    #                (20, 20)), width=10)
    return PIL_image


def distance(points):
    p0, p1 = points
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def center_point(kmin, kmax):
    return (((kmin + kmax) / 2).astype(int))


def calculate_overlap(rect1, rect2):
    # Calculate overlap area
    x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
    overlap_area = x_overlap * y_overlap
    return overlap_area


def rectangles_overlap(rect1, rect2, overlap_threshold):
    # Calculate areas of each rectangle
    area_rect1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area_rect2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

    # Calculate total area covered by both rectangles
    total_area = area_rect1 + area_rect2

    # Calculate overlap area
    overlap_area = calculate_overlap(rect1, rect2)

    # Calculate overlap percentage
    overlap_percentage = (overlap_area / total_area) * 100

    return overlap_percentage > overlap_threshold


def intersects_angle(point_g, ex, ey, top_left, bottom_right, angle):
    """
    Checks if a small angle surrounding a ray from point G with direction
    defined by ex and ey intersects the rectangle.

    Args:
        point_g: A numpy array representing point G (gx, gy).
        ex: The direction (in radians) along the horizontal axis.
        ey: The direction (in radians) along the vertical axis, defining the ray direction.
        top_left: A numpy array representing the top-left corner of the rectangle (x_min, y_min).
        bottom_right: A numpy array representing the bottom-right corner of the rectangle (x_max, y_max).
        angle: The angle (in degrees) defining the small opening around the ray.

    Returns:
        True if the angle intersects the rectangle, False otherwise.
    """

    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Direction vector of the ray
    ray_direction = np.array([ex, ey]) / np.linalg.norm([ex, ey])

    # Lines representing the sides of the rectangle
    top_line = [top_left, top_left + np.array([bottom_right[0] - top_left[0], 0])]
    right_line = [top_right := top_left + np.array([0, bottom_right[1] - top_left[1]]), bottom_right]
    bottom_line = [bottom_right, bottom_left := bottom_right + np.array([-bottom_right[0] + top_left[0], 0])]
    left_line = [bottom_left, top_left]

    # Check for intersection between ray and each line segment
    for line in [top_line, right_line, bottom_line, left_line]:
        intersection = line_intersection(point_g, ray_direction, line[0], line[1])
        if intersection is not None and is_inside_rectangle(intersection, top_left, bottom_right):
            return True

    # No intersection found with any side
    return False


def line_intersection(point_g, ray_direction, line_point1, line_point2):
    """
    Finds the intersection point between a ray and a line segment.

    This function is used as a helper to check intersection with rectangle sides.

    Args:
        point_g: The origin point of the ray (gx, gy).
        ray_direction: The direction vector of the ray.
        line_point1: The starting point of the line segment.
        line_point2: The ending point of the line segment.

    Returns:
        The intersection point if it exists within the line segment, None otherwise.
    """

    # Check for parallel lines
    denominator = np.dot(ray_direction, line_point2 - line_point1)
    if abs(denominator) < 1e-6:
        return None
    point_g = np.array([point_g[0], point_g[1]])  # Convert point_g to a NumPy array
    # Calculate intersection point parameter
    t = np.dot(line_point1 - point_g, ray_direction) / denominator

    # Check if intersection is within line segment
    if t >= 0 and t <= 1:
        return point_g + t * ray_direction
    else:
        return None


def is_inside_rectangle(point, top_left, bottom_right):
    """
    Checks if a point is inside the rectangle defined by top left and bottom right corners.

    This function is used as a helper to verify intersection points are within rectangle bounds.

    Args:
        point: A numpy array representing the point (x, y).
        top_left: A numpy array representing the top-left corner of the rectangle (x_min, y_min).
        bottom_right: A numpy array representing the bottom-right corner of the rectangle (x_max, y_max).

    Returns:
        True if the point is inside the rectangle, False otherwise.
    """

    return (point[0] >= top_left[0] and point[0] <= bottom_right[0] and
            point[1] >= bottom_right[1] and point[1] <= top_left[1])


def find_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.
    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.
    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """
    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2) ** .5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2
    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant ** .5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant ** .5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in
                                      intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(
                discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

# def slide_add(dest_arr, incoming, frame, i):
# function takes in predicted_arr,
# for idx in incoming.shape[1]:
# if idx < frame:
# dest_arr = torch.add(dest_arr[i:i+frame], incoming])


