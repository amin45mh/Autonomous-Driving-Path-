# ***************************************************************************************
# Created for the ECE-1508 Course at University of Toronto by:
# Group 30 (Nivi, Himanshu, Joseph, Amin)
#
# This code contains methods to postprocess data coming from datacollector.py. Based on input
# data, it will calculate and add a routeplan image layer, an agent history image layer, an
# ego history image layer, and appropriate target waypoints.
# ***************************************************************************************

# ***************************************************************************************
# References:
# https://github.com/deepsense-ai/carla-birdeye-view/tree/master/carla_birdeye_view 
# ***************************************************************************************

from Data import Data
import numpy as np
from math import cos, sin, radians
from cv2 import cv2 as cv
from PIL import Image
from tqdm import tqdm
from render_birdview_as_rgb import render_birdview_as_rgb
import os
from datetime import datetime

def get_transformed_xy(reference_transform, x, y, scaling_factor):
    '''
    Calculates a transformation of a Carla location into the frame of reference given by a Carla Transform.
    Also applies the scaling factor to give a scaled x and y in the frame of reference of the vehicle.

    The coordinates are changed so that forward for the reference transform is the y axis (facing upward in a 2D plane)
    '''
    translated_x = x - reference_transform.location.x 
    translated_y = y - reference_transform.location.y

    theta = radians(reference_transform.rotation.yaw + 90) # forward is facing upwards
    rotated_x = translated_x*cos(theta) + translated_y*sin(theta)
    rotated_y = -translated_x*sin(theta) + translated_y*cos(theta)
    
    scaled_x = rotated_x*scaling_factor
    scaled_y = rotated_y*scaling_factor

    return scaled_x,scaled_y

def get_untransformed_xy(reference_transform, x, y, scaling_factor):
    '''
    Inverts the calculations done in get_transformed_xy
    '''
    unscaled_x = x/scaling_factor
    unscaled_y = y/scaling_factor

    theta = -radians(reference_transform.rotation.yaw + 90)
    unrotated_x = unscaled_x*cos(theta) + unscaled_y*sin(theta)
    unrotated_y = -unscaled_x*sin(theta) + unscaled_y*cos(theta)

    untranslated_x = unrotated_x + reference_transform.location.x 
    untranslated_y = unrotated_y + reference_transform.location.y

    return untranslated_x,untranslated_y

def get_frame_xy(x_in, y_in, image_dim_px):
    '''
    Transforms a given x and y coordinate in ego's frame of reference (with y axis pointing up in the plane) into an image
    coordinate. Assumes a square image of image_dim pixels and ego being at the center. 

    Image coordinates are returned with top left as (0,0)
    '''
    frame_x = image_dim_px/2 + x_in
    frame_y = image_dim_px/2 + y_in

    return frame_x, frame_y

def get_frame_xy_inv(frame_x, frame_y, image_dim_px):
    '''
    Inverse of get_frame_xy
    '''
    x = frame_x - image_dim_px/2
    y = frame_y - image_dim_px/2

    return x, y

def get_empty_mask(image_dim_px):
    '''
    Creates an empty image layer
    '''
    return np.zeros((image_dim_px,image_dim_px))

def get_routeplan_mask(data, pixels_per_meter, image_dim_px):
    '''
    Builds a mask for the routeplan given data about the current timestep of the simulation
    '''
    routeplan_mask = get_empty_mask(image_dim_px)

    # Build a list of waypoints from the given data
    waypoint_list = []
    for val in data.ego_waypoints:
        waypoint_list.append(val[0])
    
    transformed_waypoints_in_frame = []
    for waypoint in waypoint_list:
        transform_x, transform_y = get_transformed_xy(data.ego_transform, 
                                                      waypoint.transform.location.x, 
                                                      waypoint.transform.location.y, 
                                                      pixels_per_meter)

        # Stop adding waypoints once we hit the edge of the frame
        if (abs(transform_x) >= image_dim_px/2 or abs(transform_y) >= image_dim_px/2):
            break

        frame_x, frame_y = get_frame_xy(transform_x, transform_y, image_dim_px)
        transformed_waypoints_in_frame.append((frame_x, frame_y))

    # plot waypoints on image. Only makes sense if there are 2 or more waypoints
    if len(transformed_waypoints_in_frame) > 2:
        polygon = np.array([transformed_waypoints_in_frame], dtype=np.int32)
        cv.polylines(
            img=routeplan_mask, pts=polygon, isClosed=False, color=1, thickness=1
        )
    
    return routeplan_mask

def point_transform(point, original_transform, new_transform, pixels_per_meter, image_dim_px):
    '''
    Transforms a 2D point from one image to another based on the Carla transforms of ego in those frames.
    '''
    original_local_x, original_local_y = get_frame_xy_inv(point[0], point[1], image_dim_px)
    global_x,global_y = get_untransformed_xy(original_transform, original_local_x, original_local_y, pixels_per_meter)
    new_local_x, new_local_y = get_transformed_xy(new_transform, global_x, global_y, pixels_per_meter)
    frame_x,frame_y = get_frame_xy(new_local_x, new_local_y, image_dim_px)
    return np.array([frame_x, frame_y]).astype(np.int32)

def get_history_mask(lookback_data, pixels_per_meter, image_dim_px, layer_index):
    '''
    Builds a mask for history of a given image layer given current and past data. 

    lookback_data is a list of the past n timesteps (determined external to this function), with the final item
    in the list being the frame of reference for this mask (current timestep)
    '''
    ego_transform = lookback_data[-1].ego_transform # We want to render everything in this frame
    
    output_frame = get_empty_mask(image_dim_px)

    brightness = 1
    brightness_delta = 0.05 # decrease brightness each timestep backwards
    for data in reversed(lookback_data[0:-1]):
        frame_mask = data.raw_produced[layer_index,:,:]

        # Calculate a warp matrix based on a triangle in both images, then warp the image.
        srcTri = np.array( [[0, 0], [image_dim_px - 1, 0], [0, image_dim_px - 1]] ).astype(np.float32)
        dstTri = np.array( [point_transform(srcTri[0], data.ego_transform, ego_transform, pixels_per_meter, image_dim_px), 
                            point_transform(srcTri[1], data.ego_transform, ego_transform, pixels_per_meter, image_dim_px), 
                            point_transform(srcTri[2], data.ego_transform, ego_transform, pixels_per_meter, image_dim_px)] ).astype(np.float32)
        warp_mat = cv.getAffineTransform(srcTri, dstTri)
        warp_dst = cv.warpAffine(frame_mask, warp_mat, (image_dim_px, image_dim_px), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)

        # Add the warped image to the output frame
        for index in np.argwhere(warp_dst):
            output_frame[index[0], index[1]] = max(output_frame[index[0], index[1]],brightness)

        brightness -= brightness_delta

    # Linear normalization to ensure the max value is 1
    min_val = 0
    max_val = max(np.max(output_frame), 1)
    output_frame = (output_frame - min_val) / (max_val - min_val)
    return output_frame

def get_ego_targets(lookahead_data, target_sample_frequency):
    '''
    Get targets for network output based on actual positions occupied by ego in the target lookahead time
    '''
    targets = []

    for index, data in enumerate(lookahead_data):
        # Check against sampling frequency
        if (index % target_sample_frequency) != 0:
            continue
        
        ego_location = data.ego_transform.location
        targets.append((ego_location.x, ego_location.y))

    # As mentioned in the final report, these targets should have been transformed into ego's local coordinates
    # to get appropriate output waypoints. We decided to submit with this issue to show our actual process. 
    # If continuing with the project for more time we would have made the change.
    targets = np.array(targets)
    return targets
    
def run_postprocessing(stored_data, output_root, samples_per_second, pixels_per_meter, image_dim_px, add_to_existing = False):

    images = []
    targets = []

    crop_dim_px = 224 # VGG16 input dimension

    history_timeout = 4.0 # how many seconds of history we want to incorporate into images
    target_lookahead = 4.0 # how many seconds to look ahead of the current timestep
    target_frequency = 0.5 # how often to extract a target ahead of ego

    end_buffer = target_lookahead*samples_per_second
    start_index = int(history_timeout*samples_per_second)
    end_index = int(len(stored_data) - end_buffer)

    print("Expected number of images:", int((end_index - start_index)))

    with tqdm(range(end_index - start_index), unit=" Timesteps") as data_bar:
        data_bar.set_description("Image Postprocessing")
        for idx_bar in data_bar:
            data_index = start_index + idx_bar
            data = stored_data[data_index]
            
            # Get the mask for the routeplan
            routeplan_mask = get_routeplan_mask(data, pixels_per_meter, image_dim_px)
            routeplan_mask = routeplan_mask.reshape((1,image_dim_px,image_dim_px))

            history_start = data_index - start_index
            # Get the mask for ego history        
            ego_history = stored_data[history_start:data_index+1]
            ego_history_mask = get_history_mask(ego_history, pixels_per_meter, image_dim_px, 4)
            ego_history_mask = ego_history_mask.reshape((1,image_dim_px,image_dim_px))

            # Get the mask for track history        
            track_history = stored_data[history_start:data_index+1]
            track_history_mask = get_history_mask(track_history, pixels_per_meter, image_dim_px, 3)
            track_history_mask = track_history_mask.reshape((1,image_dim_px,image_dim_px))

            # Add the new masks to the BEV image
            processed_image = np.concatenate((data.raw_produced, routeplan_mask))
            processed_image = np.concatenate((processed_image, ego_history_mask))
            processed_image = np.concatenate((processed_image, track_history_mask))

            # Move the layer axis to the last position in the array
            processed_image = np.moveaxis(processed_image, 0, 2)

            # Get the target data using lookahead
            lookahead_end_index = (data_index + int(target_lookahead*samples_per_second)) + 1
            target = get_ego_targets(stored_data[data_index: lookahead_end_index], target_frequency*samples_per_second)

            # Crop the data to the VGG16 input size - 224x244
            # Image will no longer be centered on ego, we'll place ego 25 % from the bottom of the image
            crop_top = int(image_dim_px/2 - 0.75*crop_dim_px)
            crop_bottom = int(image_dim_px/2 + 0.25*crop_dim_px)
            crop_left = int(image_dim_px/2 - 0.5*crop_dim_px)
            crop_right = int(image_dim_px/2 + 0.5*crop_dim_px)

            processed_image = processed_image[crop_top:crop_bottom, crop_left:crop_right, :]

            images.append(processed_image)
            targets.append(target)
    
    print("Created", len(images), "frames of data.")

    images = np.asarray(images)
    targets = np.asarray(targets)

    # Save the data to a numpy file depending on whether we want to add to an existing file or not
    if add_to_existing:
        save_path = os.path.join(output_root, "all_data.npz")
        if os.path.exists(save_path):
            previous_data = np.load(save_path, allow_pickle=True)
            print("Previous length:", previous_data['targets'].shape[0])
            images = np.append(previous_data['images'],images, axis = 0)
            targets = np.append(previous_data['targets'],targets, axis = 0)
        np.savez(save_path, images = images, targets = targets)
        print("Data saved to:", save_path)
    else:
        file_index = len([name for name in os.listdir(output_root)])
        save_path = os.path.join(output_root, str(file_index))
        np.savez(save_path, images = images, targets = targets)
        print("Data saved to:", save_path)
    
