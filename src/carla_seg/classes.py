"""Class IDs and visualization palette retained from the submitted demo."""

import numpy as np

CLASS_NAMES = (
    "None", "Buildings", "Fences", "Other", "Pedestrians", "Poles",
    "RoadLines", "Roads", "Sidewalks", "Vegetation", "Vehicles", "Walls",
    "TrafficSigns",
)
NUM_CLASSES = len(CLASS_NAMES)
CLASS_COLORS = np.array([
    [0, 0, 0], [70, 70, 70], [190, 153, 153], [55, 90, 80],
    [220, 20, 60], [153, 153, 153], [157, 234, 50], [128, 64, 128],
    [244, 35, 232], [107, 142, 35], [220, 220, 0], [102, 102, 156],
    [220, 200, 0],
], dtype=np.uint8)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
