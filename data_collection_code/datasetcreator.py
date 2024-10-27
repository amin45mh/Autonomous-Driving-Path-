# ***************************************************************************************
# Created for the ECE-1508 Course at University of Toronto by:
# Group 30 (Nivi, Himanshu, Joseph, Amin)
#
# This code converts .npz files saved with postprocessing into a format we can use as a dataset in pytorch.
# Each individual birds-eye-view is saved as a separate jpg file, and the targets are stored in a csv
# ***************************************************************************************

import os
from render_birdview_as_rgb import render_birdview_as_rgb
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from time import sleep

# Setup paths
file_root = os.getcwd()
output_root = os.path.join(file_root, "outputs")
tagets_filepath = os.path.join(output_root,"targets.csv")
data_filepath = os.path.join(file_root, "data_collected")

filepaths = []

for file in os.listdir(data_filepath):
    if file.endswith(".npz"):
        filepaths.append(os.path.join(data_filepath, file))

with tqdm(range(len(filepaths)), unit="files") as file_bar:

    for file_bar_index in file_bar:
        file_name = filepaths[file_bar_index]
        # Get the underlying arrays
        print("Loading data for {}...".format(file_name))
        with np.load(file_name) as data:
            images = data['images']
            targets = data['targets']
            targets = targets.reshape(targets.shape[0], targets.shape[1]*targets.shape[2])[:,0:16]
            targets_frame = pd.DataFrame(targets)
            targets_frame["file_name"] = ""

            # If the file is being created fresh, lets rename headers to be nice
            if not os.path.exists(tagets_filepath):
                for i in range(int(targets.shape[1]/2)):
                    targets_frame.rename(columns={i*2 :"wp_{}_x".format(i), (i*2)+1 : "wp_{}_y".format(i)}, inplace=True )

            if len(images) != len(targets):
                raise RuntimeError("Mismatched Sizes!")

            # For each image, convert to rgb, save as jpeg
            print("Converting Images...")
            for index, image in enumerate(images):
                associated_target = targets[index]
                file_index = len([name for name in os.listdir(output_root)])

                rgb = render_birdview_as_rgb(image)
                image = Image.fromarray(rgb)
                filename = "image_" + str(file_index) + ".jpg"
                image.save(os.path.join(output_root, filename))
                targets_frame.at[index, "file_name"] = filename

            include_header = True
            if os.path.exists(tagets_filepath):
                include_header = False
            targets_frame.to_csv(tagets_filepath, mode='a', header=include_header, index=False)

            print("Saved data for {}.".format(file_name))

        # rename so we don't extract it again
        os.rename(file_name, "{}.extracted".format(file_name))

print("All Data Extracted!")
