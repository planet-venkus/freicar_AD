#Exercise 3, Localization, Particle Filter
## How to start the node:
First go to `/home/freicar/freicar_ws/src/freicar-2020-exercises/02-01-localization-exercise/freicar_localization/bash` and execute `download_loc_bag.bash`. This downloads a rosbag that contains a prerecorded drive of a car. By using the prerecorded drive every team has the same driving-routes and the results are thus comparable.
However, you can also exchange running the bag with running the simulator (sim_base.launch).

Run `roslaunch freicar_localization evaluate.launch`. Take a look inside this launch file and see what is started.
Overall this launch file will play the rosbag, start the street sign detector and starts the particle filter. When you ctrl/C after the rosbag has finished playing a mean localization error is displayed.

If you want to start the node in "live mode" while running the simulator with the sim_base.launch file you can run:
`rosrun freicar_localization freicar_localization_node _use_lane_regression:=false`

If `use_lane_regression` (inside the evaluate.launch or while running the individual node) is set to true the filter subscribes to `/freicar_1/sim/camera/rgb/front/reg_bev` where it assumes it to be the birdseye lane regression map where every pixel is between 0 and 255. This map is then passed to the sensor_model class where you could implement your own sensor_model for the regression.  

## What you need to implement:
1. Particle initialization: Sample initial particles across the map by completing the function `void particle_filter::InitParticles()` in particle_filter.cpp
2. Motion Model: Implement the constant velocity motion model and apply motion to all particles with sampled gaussian noise. The function to complete: `void particle_filter::ConstantVelMotionModel(nav_msgs::Odometry odometry, float time_step)` in `particle_filter.cpp`
3. Sensor Model: Calculate the weight of every particle given the observed sign positions and the true sign positions from the map. Complete the function: `bool sensor_model::calculatePoseProbability(const std::vector<cv::Mat> lane_regression, const std::vector<Sign> observed_signs, std::vector<Particle>& particles, float& max_prob)` in `sensor_model.cpp`
4. Resampling: Implement ImportanceSampling and LowVarianceSampling in the functions `void particle_filter::ImportanceSampling()` and `void particle_filter::LowVarianceSampling()` in `particle_filter.cpp`.
  You need to uncomment the function that you want to use in `bool particle_filter::ObservationStep(const std::vector<cv::Mat> reg, const std::vector<Sign> observed_signs)`.

Please note all comments before the respective function definitions.
You might use the visualization functions from `ros_vis.cpp`

Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
