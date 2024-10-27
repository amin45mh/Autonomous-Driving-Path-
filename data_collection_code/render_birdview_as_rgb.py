# ***************************************************************************************
# Created for the ECE-1508 Course at University of Toronto by:
# Group 30 (Nivi, Himanshu, Joseph, Amin)
#
# This code contains information and methods to generate RGB images from birds-eye-view data.
# it is mostly copied from carla-birdeye-view, but modified slightly to add extra layers
# we created in postprocessing.
# ***************************************************************************************

# ***************************************************************************************
# References:
# https://github.com/deepsense-ai/carla-birdeye-view/tree/master/carla_birdeye_view 
# ***************************************************************************************

import numpy as np
from enum import IntEnum, auto, Enum
from typing import List

class BirdViewMasks(IntEnum):
    AGENT_HISTORY = 11
    EGO_HISTORY = 10
    ROUTPLAN = 9
    PEDESTRIANS = 8
    RED_LIGHTS = 7
    YELLOW_LIGHTS = 6
    GREEN_LIGHTS = 5
    AGENT = 4
    VEHICLES = 3
    CENTERLINES = 2
    LANES = 1
    ROAD = 0

    @staticmethod
    def top_to_bottom() -> List[int]:
        return list(BirdViewMasks)

    @staticmethod
    def bottom_to_top() -> List[int]:
        return list(reversed(BirdViewMasks.top_to_bottom()))

class RGB:
    VIOLET = (173, 127, 168)
    TEAL = (21, 209, 206),
    ORANGE = (252, 175, 62)
    ORANGE_LIGHT = (240, 208, 161)
    CHOCOLATE = (233, 185, 110)
    CHAMELEON = (138, 226, 52)
    CHAMELEON_LIGHT = (181, 237, 126)
    SKY_BLUE = (114, 159, 207)
    DIM_GRAY = (105, 105, 105)
    DARK_GRAY = (50, 50, 50)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)

RGB_BY_MASK = {
    BirdViewMasks.AGENT_HISTORY: RGB.ORANGE_LIGHT,
    BirdViewMasks.EGO_HISTORY: RGB.CHAMELEON_LIGHT,
    BirdViewMasks.ROUTPLAN: RGB.TEAL,
    BirdViewMasks.PEDESTRIANS: RGB.VIOLET,
    BirdViewMasks.RED_LIGHTS: RGB.RED,
    BirdViewMasks.YELLOW_LIGHTS: RGB.YELLOW,
    BirdViewMasks.GREEN_LIGHTS: RGB.GREEN,
    BirdViewMasks.AGENT: RGB.CHAMELEON,
    BirdViewMasks.VEHICLES: RGB.ORANGE,
    BirdViewMasks.CENTERLINES: RGB.CHOCOLATE,
    BirdViewMasks.LANES: RGB.WHITE,
    BirdViewMasks.ROAD: RGB.DIM_GRAY,
}

def render_birdview_as_rgb(processed_image):
    h, w, d = processed_image.shape
    assert d == len(BirdViewMasks)
    rgb_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    nonzero_indices = lambda arr: arr == 1

    for mask_type in BirdViewMasks.bottom_to_top():
        rgb_color = RGB_BY_MASK[mask_type]
        mask = processed_image[:, :, mask_type]
        # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
        rgb_canvas[nonzero_indices(mask)] = rgb_color
    return rgb_canvas