# ***************************************************************************************
# Created for the ECE-1508 Course at University of Toronto by:
# Group 30 (Nivi, Himanshu, Joseph, Amin)
#
# This code connects to the CARLA simulator to run a map we specify. It spawns
# a set number of agents as well as the "ego" vehicle (the agent whose behaviour)
# we want to replicate. The behaviour of other vehicles is controlled by an autopilot
# within the CARLA traffic manager. The behaviour of the ego vehicle is dictated by 
# a python agent provided as part of the CARLA package (see references). An important note
# is that we only spawn vehicle type autopilot agents as the spawning and control of pedestrians 
# was beyond the time scope of this project.
#
# In order to generate birds-eye-view images for use with our network implementation, we
# use carla_birdeye_view (see references). Images are produced with this package at a set
# interval and postprocessed to create and save data.
# ***************************************************************************************

# ***************************************************************************************
# References:
# https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/automatic_control.py
# https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/generate_traffic.py
# https://github.com/deepsense-ai/carla-birdeye-view/tree/master/carla_birdeye_view 
# ***************************************************************************************

import carla
import numpy as np
import time
from datetime import datetime
import logging
import os
from pathlib import Path
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from agents.navigation.behavior_agent import BehaviorAgent 
from Data import Data
from PIL import Image
from postprocessing import run_postprocessing

# ==============================================================================
# Global variables
# ==============================================================================

scenario_map = 'Town10HD'
render_mode = True
num_agents = 100 # not including ego
agents_filter = "vehicle.*"
ego_filter = "vehicle.lincoln.mkz_2020"
output_root = os.path.join(os.getcwd(),"data_collected")
Path(output_root).mkdir(parents=True, exist_ok=True)
image_dim_px = 400
pixels_per_meter= 3
fixed_delta_seconds = 0.025

run_time_s = 38 # how long to run the simulation in each spin (30 seconds + 4 second start and end buffer)
num_spins = 340 # the number of spins to do.
# ==============================================================================
# Scenario Setup
# ==============================================================================

# Access the client
client = carla.Client('localhost', 2000)
client.set_timeout(60.0)

# Set the map to one we pick. For now using Town3
world = client.load_world(scenario_map)
settings = world.get_settings()
settings.fixed_delta_seconds = fixed_delta_seconds
world.apply_settings(settings)

# Setup for ego and actors
blueprint_library = world.get_blueprint_library()
agent_blueprints = blueprint_library.filter(agents_filter)
spawn_points = world.get_map().get_spawn_points()
traffic_manager = client.get_trafficmanager()
traffic_manager.set_global_distance_to_leading_vehicle(4.0)
traffic_manager.set_respawn_dormant_vehicles(True)
traffic_manager.set_synchronous_mode(False)

birdview_producer = BirdViewProducer(
    client,  # carla.Client
    target_size=PixelDimensions(width=image_dim_px, height=image_dim_px),
    pixels_per_meter=pixels_per_meter,
    crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
    )

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor

for spin in range(num_spins):
    print ("===========", "Spin ", spin+1, "===========")
    
    vehicles_list = []
    
    print ("Setting up...")
    if num_agents + 1 < len(spawn_points):
        np.random.shuffle(spawn_points)
    elif num_agents + 1 > len(spawn_points):
        print("Unable to generate", num_agents, "agents as there are only", len(spawn_points), "spawn points in the map!")
        raise ValueError("Num agents greater than num spawn points")

    # Setup ego
    blueprint = blueprint_library.find(ego_filter)
    blueprint.set_attribute('role_name', 'hero')
    if blueprint.has_attribute('color'):
        color = np.random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)

    # Check that we aren't spawning ego in an intersection. Loop until we find a non-junction spawn.
    ego_spawn_index = 0
    if world.get_map().get_waypoint(spawn_points[ego_spawn_index].location).is_junction:
        ego_spawn_index+=1

    ego_vehicle = world.spawn_actor(blueprint,spawn_points[0]) # spawn ego at the first spawn point
    tm_port = traffic_manager.get_port()
    vehicles_list.append(ego_vehicle.id)

    # Setup all actors
    batch = []
    for n, transform in enumerate(spawn_points):
        if n == ego_spawn_index:
            # Don't spawn on ego's spawn
            continue
        if n >= num_agents:
            break
        blueprint = np.random.choice(agent_blueprints)
        if blueprint.has_attribute('color'):
            color = np.random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = np.random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        else:
            blueprint.set_attribute('role_name', 'autopilot')

        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, True):
        if response.error:
            print(response.error)
        else:
            vehicles_list.append(response.actor_id)
        
    # Create ego behaviour
    agent = BehaviorAgent(ego_vehicle, behavior="cautious")
    destination = np.random.choice(spawn_points).location # choose random destination
    agent.set_destination(destination)

    start_time = 0
    stored_data = []

    ticks = 0
    samples_per_second = 10
    ticks_per_sample = 1/fixed_delta_seconds/samples_per_second
    print ("Running...")
    while ticks <= run_time_s/fixed_delta_seconds:
        world.wait_for_tick()

        if agent.done():
            agent.set_destination(np.random.choice(spawn_points).location)
            print("The target has been reached, searching for another target")
        
        control = agent.run_step()
        control.manual_gear_shift = False
        ego_vehicle.apply_control(control)
        
        if ticks % ticks_per_sample == 0:
            birdview = birdview_producer.produce(agent_vehicle=ego_vehicle)
            rgb = BirdViewProducer.as_rgb(birdview)
            im = Image.fromarray(rgb)
            im.save("test_birdview.jpeg")
            stored_data.append(Data(birdview, agent.get_global_plan(), ego_vehicle.get_transform()))
        
        ticks += 1

    print('\nDestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    print("Stored data length:", len(stored_data))

    print("Running postprocessing on data and saving...")
    run_postprocessing(stored_data, output_root, samples_per_second, pixels_per_meter, image_dim_px, False)

print ("===========", "Done ", "===========")