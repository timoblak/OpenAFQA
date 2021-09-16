from skimage.morphology import skeletonize
from afqa_toolbox.tools import create_minutiae_record
from skimage.color import label2rgb
import numpy as np
import cv2

NBHD = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
CONN4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]
CONN8 = [(1, 1), (1, -1), (-1, -1), (-1, 1)]


def compute_cn(window, label, return_location=False):
    """
    The function which computes the crossing number of an image window.
    The algorithm always starts and ends at the upper left corner of the window.
    This version works for windows of arbitrary size, the CN is always computed at window border pixels

    :param window: Window where zeroes are background and numbers > 0 are foreground (friction ridge)
    :param label: Label for which the CN should be computed
    :param return_location: flag whether to return the location of the last crossing - useful for determining minutia angle
    :return:  The crossing number
    """
    h, w = window.shape
    loc = np.array([0, 0])
    binary_window = window.copy()
    binary_window[binary_window > 0] = 1
    moves = [(0, 1)] * (w-1) + [(1, 0)] * (h-1) + [(0, -1)] * (w-1) + [(-1, 0)] * (h-1)
    last_observed_crossing = None

    # Go clockwise from upper left corner and count number of crossings.
    # Count only if one of crossing pixels matches with the label
    cn = 0
    for move in moves:
        if window[loc[0], loc[1]] == label or window[loc[0] + move[0], loc[1] + move[1]] == label:
            cn += np.abs(binary_window[loc[0], loc[1]] - binary_window[loc[0] + move[0], loc[1] + move[1]])
            last_observed_crossing = (loc[0], loc[1])
        loc += move

    if return_location:
        return cn / 2, last_observed_crossing
    return cn / 2


def trace_one(window, painted, c, label):
    """A recursive function, which traces a skeletonized ridge from location c until the end of ridge or image border.
    The function returns nothing, but modifies the contents of "painted".

    :param window: A local window of a binary skeletonized friction ridge image
    :param painted: A the same local window, where each individual traced ridge has a label associated to it
    :param c: the current location inside the local window
    :param label: the label for which the ridge is traced
    """
    h, w = window.shape
    painted[c[0], c[1]] = label

    for d in CONN4:
        dx = c[0] + d[0]
        dy = c[1] + d[1]
        if dx < 0 or dx >= w or dy < 0 or dy >= h:
            continue
        if window[dx, dy] and painted[dx, dy] == 0:
            trace_one(window, painted, (dx, dy), label)

    for d in CONN8:
        dx = c[0] + d[0]
        dy = c[1] + d[1]
        if dx < 0 or dx >= w or dy < 0 or dy >= h:
            continue
        if window[dx, dy] and painted[dx, dy] == 0:
            trace_one(window, painted, (dx, dy), label)


def trace_ridges(window):
    """
    This function traces the ridges within a local window from a minutia point to rodge endings.
    It turns a binary skeletonized ridge structure and segments it into individual ridges, like so (X is the detected minutia):
    0 0 0 0 0 0 0       0 0 0 0 0 0 0
    1 0 0 0 0 0 1       1 0 0 0 0 0 2
    0 1 1 0 1 1 0       0 1 1 0 2 2 0
    0 0 1 1 0 0 0   >   0 0 1 X 0 0 0
    0 0 0 1 0 0 0       0 0 0 3 0 0 0
    0 0 0 0 1 0 0       0 0 0 0 3 0 0
    0 0 0 0 0 1 1       0 0 0 0 0 3 3

    This way, we can verify the existance of a minutia point and we can calculate its angle


    :param window: A local window of a binary skeletonized friction ridge image.
    :return: The same local window, where each traced ridge has its own label associated to it.
    """
    c = window.shape[0]//2
    painted = np.zeros(window.shape)
    painted[c, c] = -1
    label = 1

    # First, vertical end horizontal neighbors are traced to prevent early stopping of tracing
    starts = {}
    for d in CONN4:
        if window[c+d[0], c+d[1]] and painted[c+d[0], c+d[1]] == 0:
            painted[c+d[0], c+d[1]] = label
            starts[label] = c+d[0], c+d[1]
            label += 1
    for start_label in starts:
        trace_one(window, painted, starts[start_label], start_label)

    # Then, diagonal neighbors are traced
    starts = {}
    for d in CONN8:
        if window[c+d[0], c+d[1]] and painted[c+d[0], c+d[1]] == 0:
            painted[c + d[0], c + d[1]] = label
            starts[label] = c + d[0], c + d[1]
            label += 1
    for start_label in starts:
        trace_one(window, painted, starts[start_label], start_label)

    return painted


def calculate_angle(c, crossing):
    """Calculate the angle between line from center to crossing and x axis"""
    # Negate y axis, since the y axis is inverted
    dy = -crossing[0] + c[0]
    dx = crossing[1] - c[1]
    angle = np.arctan2(dy, dx)
    angle = angle + np.pi * 2 if angle < 0 else angle
    return angle


def smallest_arc(angle1, angle2):
    """Calculates the smallest arc between two angles"""
    arc1 = np.abs(angle1 - angle2)
    arc2 = (2 * np.pi - np.max([angle1, angle2])) + np.min([angle1, angle2])
    return np.min([arc1, arc2])


def angle_data_mean(angles):
    """Calculates the mean angle of a list of angles (in radians, + inverted y axis)"""
    xs = [np.cos(a) for a in angles[::-1]]
    ys = [-np.sin(a) for a in angles[::-1]]
    angle = np.arctan2(np.sum(ys), np.sum(xs))
    if angle < 0:
        angle += 2 * np.pi
    return 2 * np.pi - angle


class MinutiaeExtraction:
    """The class for minutiae extraction using the Crossing Number method.

    Algorithm autline:
    - The method receives as input a binarized friction ridge image, where

    """
    def __init__(self):
        self.min_dist = 10
        self.placeholder_quality = 60

    def initial_detection(self, binarized_image):

        skeleton = skeletonize(binarized_image).astype(int)

        ridges = np.where(skeleton > 0)
        minutiae = []

        for k, (y, x) in enumerate(zip(*ridges)):
            # Determine crossing number
            cn = compute_cn(skeleton[y - 1:y + 2, x - 1: x + 2], 1)

            if cn in [1, 3]:
                # minutia type: 1 for ending, 2 for bifurcation
                win_radius = 5
                local_window = skeleton[y - win_radius:y + win_radius + 1, x - win_radius:x + win_radius + 1]
                cx, cy = local_window.shape[1] // 2, local_window.shape[0] // 2
                painted = trace_ridges(local_window)

                # A placeholder value is used for quality.
                # Use your own heuristics/methods to determine the quality of individual minutiae
                quality = self.placeholder_quality

                if cn == 1:
                    min_type = 1
                    # If ENDING, there should only be one ridge going from the minutia to the local image border
                    ridge1, loc = compute_cn(painted, 1, return_location=True)
                    if ridge1 != 1:
                        continue

                    # Calculate the angle of the minutia.
                    # > Angle of line between center and ending of ridge
                    angle = calculate_angle((cx, cy), loc)
                else:
                    min_type = 2
                    # If BIFURCATION, there should be three ridges going from the minutia to the local image border
                    ridge1, loc1 = compute_cn(painted, 1, return_location=True)
                    ridge2, loc2 = compute_cn(painted, 2, return_location=True)
                    ridge3, loc3 = compute_cn(painted, 3, return_location=True)
                    if ridge1 != 1 or ridge2 != 1 or ridge3 != 1:
                        continue

                    # Calculate the angle of the minutia:
                    # > Calculate angles 3 line connecting from center to endings of the three ridges
                    # > determined angles with smalled difference between them
                    # > minutia angle is then the average of these two ridge angles
                    angle1 = calculate_angle((cx, cy), loc1)
                    angle2 = calculate_angle((cx, cy), loc2)
                    angle3 = calculate_angle((cx, cy), loc3)

                    # Default angle is mean between ridge1 and ridge2
                    angle = angle_data_mean([angle1, angle2])
                    min_arc = smallest_arc(angle1, angle2)
                    # If angle between ridge1 and ridge3 is smaller, use it
                    if (new_smallest := smallest_arc(angle1, angle3)) < min_arc:
                        angle = angle_data_mean([angle1, angle3])
                        min_arc = new_smallest
                    # If angle between ridge2 and ridge3 is smaller, use it
                    if smallest_arc(angle2, angle3) < min_arc:
                        angle = angle_data_mean([angle2, angle3])

                # Convert and quantize radians to range 0-256
                angle = int((angle / (2 * np.pi)) * 256)
                minutiae.append((x, y, min_type, angle, quality))
        return minutiae

    def distance_filter(self, minutiae):
        to_remove = set()
        for i in range(1, len(minutiae)):
            for j in range(0, i):
                dist = np.sqrt((minutiae[j][0] - minutiae[i][0]) ** 2 + (minutiae[j][1] - minutiae[i][1]) ** 2)
                if dist < self.min_dist:
                    to_remove.add(i)
                    to_remove.add(j)
        filtered_minutiae = []
        for i in range(1, len(minutiae)):
            if i not in to_remove:
                filtered_minutiae.append(minutiae[i])
        return filtered_minutiae

    def extract(self, binarized_image):
        initial_minutiae = self.initial_detection(binarized_image)
        distance_filtered_minutiae = self.distance_filter(initial_minutiae)
        return create_minutiae_record(binarized_image.shape, distance_filtered_minutiae)




