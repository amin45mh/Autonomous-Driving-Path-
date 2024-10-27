This folder contains the code used for ECE1508: Applied Deep Learning by Group 30. 

## Organization:
* data_collection_code: contains all code to generate data using the CARLA simulator. Requires that you have the CARLA simulator installed.
* model_implementation_training_and_testing: contains python notebook used to define our CNN, load the dataset, train the network, and test.

## Data Collection Usage:
1. Run the CARLA simulator server - follow steps in https://carla.readthedocs.io/en/latest/start_quickstart/ to install and run.
2. Edit the datacollector.py script with the parameters you would like (map, seconds per spin, number of spins, etc)
3. Run the datacollector.py script which will run a simulation, perform post processing using postprocessing.py and save data in numpy format into a folder.
4. To extract the dataset, modify and run datasetcreator.py, which will compile birds-eye-view images and add targets and associated image paths to a csv file.

## Final Dataset Links (used in project):
Full Dataset on Town 3 - https://drive.google.com/file/d/1-6LLazewEM7Z7VP_6oPY1_NVFVOlgUVt/view?usp=drive_link
Test Dataset on Town 10 for map generalization - https://drive.google.com/file/d/1M82RrcvWkvOmWjNUGXJZ77Drpaz20mtO/view?usp=drive_link

## Notebook Usage:
1. Compress the data created above into a zip file and upload to colab or environment of your choice.
2. Edit the ECE 1508 Project python notebook to switch out things that are specific to your running environment
3. Run notebook cells as needed (note that some cells do not need to be run every time and some cells should be modified and run multiple times for iterative training)

## Attribution
The entire folder titled 'agents' is directly copied from the CARLA simulator github (https://github.com/carla-simulator/carla/tree/master/PythonAPI/carla/agents) as it contains the code used to generate an autopilot planner for the autonomous vehicle.