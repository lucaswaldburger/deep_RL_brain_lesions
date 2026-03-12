import logging

import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Actions
MOVE_RIGHT = 0
MOVE_DOWN = 1
SCALE_UP = 2
ASPECT_RATIO_UP = 3
MOVE_LEFT = 4
MOVE_UP = 5
SCALE_DOWN = 6
ASPECT_RATIO_DOWN = 7
SPLIT_HORIZONTAL = 8
SPLIT_VERTICAL = 9
PLACE_LANDMARK = 10

DIMENSION = DEFAULT_CONFIG.dimension
STEP_FACTOR = DEFAULT_CONFIG.step_factor
MAX_ASPECT_RATIO = DEFAULT_CONFIG.max_aspect_ratio
MIN_ASPECT_RATIO = DEFAULT_CONFIG.min_aspect_ratio
MIN_BOX_SIDE = DEFAULT_CONFIG.min_box_side


class ObjLocaliser:
    """Object localizer agent that uses a sliding window approach."""

    def __init__(self, image, boundingBoxes):
        resized_img = cv.resize(image, (DIMENSION, DIMENSION))
        self.image_playground = np.array(resized_img)
        self.yscale = float(DIMENSION) / image.shape[0]
        self.xscale = float(DIMENSION) / image.shape[1]
        self.targets = self._prepare_targets(boundingBoxes)
        self.agent_window = np.array([0, 0, DIMENSION, DIMENSION])
        self.iou = 0

    def Reset(self, image):
        """Reset the agent window to prepare for a new episode.

        Args:
            image: The image the agent will interact with.
        """
        self.agent_window = np.array([0, 0, DIMENSION, DIMENSION])
        resized_img = cv.resize(image, (DIMENSION, DIMENSION))
        self.image_playground = np.array(resized_img)

    def _prepare_targets(self, boundingBoxes):
        """Load and scale bounding boxes to [xmin, ymin, xmax, ymax] format.

        Args:
            boundingBoxes: A dictionary of bounding boxes.

        Returns:
            A list of scaled bounding boxes.
        """
        numOfObj = len(boundingBoxes['xmax'])
        objs = []
        for i in range(numOfObj):
            temp = [
                boundingBoxes['xmin'][i] * self.xscale,
                boundingBoxes['ymin'][i] * self.yscale,
                boundingBoxes['xmax'][i] * self.xscale,
                boundingBoxes['ymax'][i] * self.yscale,
            ]
            objs.append(temp)
        return objs

    def wrapping(self):
        """Resize the current agent window to (DIMENSION, DIMENSION).

        Returns:
            The resized current window.
        """
        im2 = self.image_playground[
            self.agent_window[1]:self.agent_window[3],
            self.agent_window[0]:self.agent_window[2],
        ]
        return cv.resize(im2, (DIMENSION, DIMENSION))

    def takingActions(self, action):
        """Perform an action and compute the reward.

        Args:
            action: Action to take.

        Returns:
            Reward corresponding to the taken action.
        """
        newbox = np.array([0, 0, 0, 0])
        termination = False

        action_map = {
            MOVE_RIGHT: self.MoveRight,
            MOVE_DOWN: self.MoveDown,
            SCALE_UP: self.scaleUp,
            ASPECT_RATIO_UP: self.aspectRatioUp,
            MOVE_LEFT: self.MoveLeft,
            MOVE_UP: self.MoveUp,
            SCALE_DOWN: self.scaleDown,
            ASPECT_RATIO_DOWN: self.aspectRatioDown,
            SPLIT_HORIZONTAL: self.splitHorizontal,
            SPLIT_VERTICAL: self.splitVertical,
            PLACE_LANDMARK: self.placeLandmark,
        }

        if action in action_map:
            newbox = action_map[action]()
        if action == PLACE_LANDMARK:
            termination = True

        self.agent_window = newbox
        self.adjustAndClip()
        r, new_iou = self.ComputingReward(self.agent_window, termination)
        self.iou = new_iou
        return r

    def MoveRight(self):
        """Action: move right. Preserves box width and height."""
        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        step = STEP_FACTOR * boxW
        if newbox[2] + step < self.image_playground.shape[0]:
            newbox[0] += step
            newbox[2] += step
        else:
            newbox[0] = self.image_playground.shape[0] - boxW - 1
            newbox[2] = self.image_playground.shape[0] - 1
        return newbox

    def MoveDown(self):
        """Action: move down. Preserves box width and height."""
        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        step = STEP_FACTOR * boxH
        if newbox[3] + step < self.image_playground.shape[1]:
            newbox[1] += step
            newbox[3] += step
        else:
            newbox[1] = self.image_playground.shape[1] - boxH - 1
            newbox[3] = self.image_playground.shape[1] - 1
        return newbox

    def scaleUp(self):
        """Action: scale up. Preserves aspect ratio."""
        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        boxH = newbox[3] - newbox[1]

        widthChange = STEP_FACTOR * boxW
        heightChange = STEP_FACTOR * boxH

        if boxW + widthChange < self.image_playground.shape[0]:
            if boxH + heightChange < self.image_playground.shape[1]:
                newDelta = STEP_FACTOR
            else:
                newDelta = self.image_playground.shape[1] / boxH - 1
        else:
            newDelta = self.image_playground.shape[0] / boxW - 1
            if boxH + (newDelta * boxH) >= self.image_playground.shape[1]:
                newDelta = self.image_playground.shape[1] / boxH - 1

        widthChange = newDelta * boxW / 2.0
        heightChange = newDelta * boxH / 2.0
        newbox[0] -= widthChange
        newbox[1] -= heightChange
        newbox[2] += widthChange
        newbox[3] += heightChange
        return newbox

    def aspectRatioUp(self):
        """Action: increase aspect ratio. Preserves width."""
        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        boxW = newbox[2] - newbox[0]

        heightChange = STEP_FACTOR * boxH

        if boxH + heightChange < self.image_playground.shape[1]:
            ar = (boxH + heightChange) / boxW
            newDelta = STEP_FACTOR if ar < MAX_ASPECT_RATIO else 0.0
        else:
            newDelta = self.image_playground.shape[1] / boxH - 1
            ar = (boxH + newDelta * boxH) / boxW
            if ar > MAX_ASPECT_RATIO:
                newDelta = 0.0

        heightChange = newDelta * boxH / 2.0
        newbox[1] -= heightChange
        newbox[3] += heightChange
        return newbox

    def MoveLeft(self):
        """Action: move left. Preserves box width and height."""
        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        step = STEP_FACTOR * boxW
        if newbox[0] - step >= 0:
            newbox[0] -= step
            newbox[2] -= step
        else:
            newbox[0] = 0
            newbox[2] = boxW
        return newbox

    def MoveUp(self):
        """Action: move up. Preserves box width and height."""
        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        step = STEP_FACTOR * boxH
        if newbox[1] - step >= 0:
            newbox[1] -= step
            newbox[3] -= step
        else:
            newbox[1] = 0
            newbox[3] = boxH
        return newbox

    def scaleDown(self):
        """Action: scale down. Preserves aspect ratio."""
        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        boxW = newbox[2] - newbox[0]

        widthChange = STEP_FACTOR * boxW
        heightChange = STEP_FACTOR * boxH

        if boxW - widthChange >= MIN_BOX_SIDE:
            if boxH - heightChange >= MIN_BOX_SIDE:
                newDelta = STEP_FACTOR
            else:
                newDelta = MIN_BOX_SIDE / boxH - 1
        else:
            newDelta = MIN_BOX_SIDE / boxW - 1
            if boxH - newDelta * boxH < MIN_BOX_SIDE:
                newDelta = MIN_BOX_SIDE / boxH - 1

        widthChange = newDelta * boxW / 2.0
        heightChange = newDelta * boxH / 2.0
        newbox[0] += widthChange
        newbox[1] += heightChange
        newbox[2] -= widthChange
        newbox[3] -= heightChange
        return newbox

    def splitHorizontal(self):
        """Action: horizontal splitting."""
        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        if boxW > MIN_BOX_SIDE:
            newbox[2] -= boxW / 2.0
        return newbox

    def splitVertical(self):
        """Action: vertical splitting."""
        newbox = np.copy(self.agent_window)
        boxH = newbox[3] - newbox[1]
        if boxH > MIN_BOX_SIDE:
            newbox[3] -= boxH / 2.0
        return newbox

    def aspectRatioDown(self):
        """Action: decrease aspect ratio. Preserves height."""
        newbox = np.copy(self.agent_window)
        boxW = newbox[2] - newbox[0]
        boxH = newbox[3] - newbox[1]

        widthChange = STEP_FACTOR * boxW
        if boxW + widthChange < self.image_playground.shape[0]:
            ar = boxH / (boxW + widthChange)
            newDelta = STEP_FACTOR if ar >= MIN_ASPECT_RATIO else 0.0
        else:
            newDelta = self.image_playground.shape[0] / boxW - 1
            ar = boxH / (boxW + newDelta * boxW)
            if ar < MIN_ASPECT_RATIO:
                newDelta = 0.0

        widthChange = newDelta * boxW / 2.0
        newbox[0] -= widthChange
        newbox[2] += widthChange
        return newbox

    def placeLandmark(self):
        """Termination action. Draws a cross on the image to mark search end."""
        newbox = np.copy(self.agent_window)
        h = int((newbox[3] - newbox[1]) / 2)
        h_l = int(h / 5)
        w = int((newbox[2] - newbox[0]) / 2)
        w_l = int(w / 5)

        self.image_playground[
            newbox[1] + h - h_l:newbox[1] + h + h_l,
            newbox[0]:newbox[2],
        ] = 0
        self.image_playground[
            newbox[1]:newbox[3],
            newbox[0] + w - w_l:newbox[0] + w + w_l,
        ] = 0
        return newbox

    def adjustAndClip(self):
        """Clip the agent window to image boundaries."""
        w = self.agent_window
        shape = self.image_playground.shape

        # Clip x of top-left corner
        if w[0] < 0:
            step = -w[0]
            if w[2] + step < shape[0]:
                w[0] += step
                w[2] += step
            else:
                w[0] = 0
                w[2] = shape[0] - 1

        # Clip y of top-left corner
        if w[1] < 0:
            step = -w[1]
            if w[3] + step < shape[1]:
                w[1] += step
                w[3] += step
            else:
                w[1] = 0
                w[3] = shape[1] - 1

        # Clip x of bottom-right corner
        if w[2] >= shape[0]:
            step = w[2] - shape[0]
            if w[0] - step >= 0:
                w[0] -= step
                w[2] -= step
            else:
                w[0] = 0
                w[2] = shape[0] - 1

        # Clip y of bottom-right corner
        if w[3] >= shape[1]:
            step = w[3] - shape[1]
            if w[1] - step >= 0:
                w[1] -= step
                w[3] -= step
            else:
                w[1] = 0
                w[3] = shape[1] - 1

        # Ensure minimum box size
        if w[0] == w[2]:
            if w[2] + MIN_BOX_SIDE < shape[0]:
                w[2] = w[2] + MIN_BOX_SIDE
            else:
                w[0] = w[0] - MIN_BOX_SIDE

        if w[1] == w[3]:
            if w[3] + MIN_BOX_SIDE < shape[1]:
                w[3] = w[3] + MIN_BOX_SIDE
            else:
                w[1] = w[1] - MIN_BOX_SIDE

    def intersectionOverUnion(self, boxA, boxB):
        """Compute IoU between two boxes.

        Args:
            boxA: First box [xmin, ymin, xmax, ymax].
            boxB: Second box [xmin, ymin, xmax, ymax].

        Returns:
            IoU value.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def ComputingReward(self, agent_window, termination=False):
        """Compute the reward for a given action by checking all ground truths.

        Args:
            agent_window: Current agent window.
            termination: Whether this is the termination action.

        Returns:
            Tuple of (reward, max_iou).
        """
        max_iou = -2
        reward = 0
        for target in self.targets:
            new_iou = self.intersectionOverUnion(agent_window, np.array(target))
            if new_iou > max_iou:
                max_iou = new_iou
                reward = self.ReturnReward(new_iou, termination)
        if termination:
            max_iou = 0
        return reward, max_iou

    def ReturnReward(self, new_iou, termination):
        """Compute reward based on IoU change and termination.

        Args:
            new_iou: IoU for the current action.
            termination: Whether this is the termination action.

        Returns:
            Reward value.
        """
        reward = 1 if new_iou - self.iou > 0 else -1

        if termination:
            reward = 3 if new_iou > DEFAULT_CONFIG.iou_threshold else -3
            self.iou = 0
        return reward

    def drawActions(self):
        """Display the image with bounding boxes and agent window."""
        fig, ax = plt.subplots(1)
        ax.imshow(self.image_playground)

        rect = patches.Rectangle(
            (self.agent_window[0], self.agent_window[1]),
            self.agent_window[2] - self.agent_window[0],
            self.agent_window[3] - self.agent_window[1],
            linewidth=1, edgecolor='r', facecolor='none',
        )
        ax.add_patch(rect)

        for target in self.targets:
            rect2 = patches.Rectangle(
                (target[0], target[1]),
                target[2] - target[0],
                target[3] - target[1],
                linewidth=1, edgecolor='b', facecolor='none',
            )
            ax.add_patch(rect2)

        plt.draw()
        plt.show()

    def my_draw(self):
        """Render current state as an image array (for animation)."""
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        ax.imshow(self.image_playground)

        rect = patches.Rectangle(
            (self.agent_window[0], self.agent_window[1]),
            self.agent_window[2] - self.agent_window[0],
            self.agent_window[3] - self.agent_window[1],
            linewidth=1, edgecolor='r', facecolor='none',
        )
        ax.add_patch(rect)

        for target in [self.targets[0]]:
            rect2 = patches.Rectangle(
                (target[0], target[1]),
                target[2] - target[0],
                target[3] - target[1],
                linewidth=1, edgecolor='b', facecolor='none',
            )
            ax.add_patch(rect2)

        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        return np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
