# ***************************************************************************************
# Created for the ECE-1508 Course at University of Toronto by:
# Group 30 (Nivi, Himanshu, Joseph, Amin)
#
# This code contains the class of data used for saving and passing BEV images to postprocessing
# ***************************************************************************************

from dataclasses import dataclass
import numpy as np
import carla

@dataclass
class Data:
    "Class for keeping produced data for postprocessing"
    raw_produced: np.ndarray
    ego_waypoints: list
    ego_transform: carla.Transform