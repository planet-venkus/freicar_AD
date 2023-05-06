#!/bin/bash
# Setup script for setting environment for team roboracers project

OBJECT_DETECTION=/home/freicar/freicar_ws/src/freicar-2020-exercises/01-01-object-detection-roboracers/ROS/image_boundingboxinfo_publisher/script/


source /opt/conda/etc/profile.d/conda.sh
conda activate freicar
echo "Conda activated"

echo "freicar environment activated"

cd $OBJECT_DETECTION 
python image_pub_bbs.py $1 $2

echo "object detection node started!"
