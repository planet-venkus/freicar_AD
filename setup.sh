#!/bin/bash
# Setup script for setting environment for team roboracers project

LOCALIZATION_DIR=/home/freicar/freicar_ws/src/freicar-2020-exercises/02-01-localization-roboracers/freicar_localization/bash/
WORKSPACE_DIR=/home/freicar/freicar_ws
EXERCISE_DIR=/home/freicar/freicar_ws/src/freicar-2020-exercises
OBJECT_DETECTION_DIR=/home/freicar/freicar_ws/src/freicar-2020-exercises/01-01-object-detection-roboracers/ROS/image_boundingboxinfo_publisher/script

cd $LOCALIZATION_DIR

echo "changed directory to:"
echo $PWD

echo "Downloading ros bag, please wait .... "
./download_loc_bag.bash

echo "Change dir"
cd $WORKSPACE_DIR

echo $PWD

echo "Cleaning ...."
catkin clean --all

echo "Building localization node first ..."
catkin build freicar_localization_rr

echo "Building ..."
catkin build

echo "Change dir"
cd $EXERCISE_DIR
echo $PWD

source /opt/conda/etc/profile.d/conda.sh
conda activate freicar
echo "Installing packages for python ..."
pip install -r requirements.txt
conda deactivate

echo "Change dir"
cd $OBJECT_DETECTION_DIR
echo $PWD

chmod +x image_pub_bbs.sh

echo "Setup done!"
