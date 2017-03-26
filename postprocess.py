import numpy as np
import cv2
from scipy.ndimage.measurements import label

class RingBufSmoother(object):
    """
    Smoothes heatmaps across several iterations and applies thresholds
    """

    def __init__(self, shape, length=10, threshold=4):
        self.length = length
        self.data = np.zeros([length] + list(shape), dtype=np.float32)
        self.threshold = threshold
        self.index = 0
        self.count = 0

    def extend(self, x):
        """
        Adds array x to ring buffer.
        :param x: The element to add to the RingBuffer
        """
        self.data[self.index] = x
        self.index = (self.index + 1) % self.length

        self.count += 1
        if self.count > len(self.data):
            self.count = len(self.data)

    def mean(self):
        return np.mean(self.data[:self.count], axis = 0)

    def rolling_threshold(self):
        heatmap = self.mean() 
        heatmap[heatmap < self.threshold] = 0
        return heatmap

def segment_cars(heatmaps):
    """
    Get a map of where each car is located.
    :param heatmaps: The car detection heatmaps to use for segmentation.
    :return: A tuple (segmentation maps, num_cars).
    """
    segmentation_maps = np.empty_like(heatmaps)
    num_cars = []

    for i, heatmap in enumerate(heatmaps):
        frame_label = label(heatmap)
        segmentation_maps[i] = frame_label[0]
        num_cars.append(frame_label[1])

    return segmentation_maps, num_cars


def draw_boxes(imgs, segmentation_maps, num_cars):
    """
    Draw bounding boxes around each car in the images.
    :param imgs: The original frames.
    :param segmentation_maps: The segmentation maps of cars in each frame.
    :param num_cars: The number of cars detected in each frame.
    :return: The images, superimposed with bounding boxes.
    """
    imgs_superimposed = imgs.copy()
    overlays = imgs.copy()
    for i, img in enumerate(imgs):
        for car_num in range(1, num_cars[i] + 1):
            # Find pixels with each car_number label value
            nonzero = np.where(segmentation_maps[i] == car_num)

            # Identify x and y values of those pixels
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            box = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))

            # Draw the box on the image
            color = (228, 179, 0)
            cv2.rectangle(imgs_superimposed[i], box[0], box[1], color, 2)
            cv2.rectangle(overlays[i], box[0], box[1], color, -1)

        # cv2.addWeighted(imgs_superimposed[i], 0.8, overlays[i], 0.2, 0)
        imgs_superimposed[i] = cv2.addWeighted(overlays[i], 0.5, imgs_superimposed[i], 0.5, 0)

    return imgs_superimposed

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img