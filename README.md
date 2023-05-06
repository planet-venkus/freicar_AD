# Team Roboracers

## Below steps can be used for running the project:
* Clone the repo and put the contents inside freicar-2020-exercises folder or clone the repo directly in freicar-2020-exercises folder
* Run the setup.sh file with the command ./setup.sh
* Run `roslaunch freicar_launch local_comp_launch.launch`

## Launch File:
* Run command `roslaunch freicar_race.launch` by going inside freicar-2020-exercises folder

## Other Config:
### For rviz, use the config file provided in this repo:
* rviz_conf.rviz

### For changing the map, change in following locations:
* Inside local_comp_launch.launch file
* Inside /home/freicar/freicar_ws/src/freicar_base/freicar_setting/param/freicar_settings.yaml
* Inside /home/freicar/freicar_ws/src/freicar-2020-exercises/freicar_race.launch file
* Inside /home/freicar/freicar_ws/src/freicar-2020-exercises/02-01-localization-roboracers/freicar_localization/launch/freicar_localization.launch file

### For comp mode, change in the file below:
* freicar-base/freicar-launch/launch/sim-base.launch

## Manual Steps below(If not using launch file):
* roscore
* Run simulator with roslaunch freicar_launch sim_base.launch
* rosservice call /freicar_1/set_position "{x: 0.3, y: 1.0, z: 0.0, heading: 0.0}"
* rosrun freicar_localization_rr freicar_localization_rr_node
* rosrun freicar_sign_detect freicar_sign_detect_node
* roslaunch freicar_control_rr start_controller.launch
* rosrun freicar_drive freicar_drive_node
* For testing, freicar_race2.launch is give. Run sim-base with that launch file. For changing the map, change the map location in Inside /home/freicar/freicar_ws/src/freicar_base/freicar_launch/launch/sim_base.launch


Contributors:
1. Venkat Subramanyam
2. Karan Pathak
3. Jayanth Sharma
4. Amith B G
5. Nayana Koneru