/*
 * Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
 * Project: FreiCAR
 * Do NOT distribute this code to anyone outside the FreiCAR project
 */

#include "controller.h"
#include <yaml-cpp/yaml.h>


/*
 * Executes one step of the PID controller
 */
float PID::step(const float error, const ros::Time stamp){
    ros::Duration dt = stamp - prev_t;

    double delta_e = (error - prev_e) / dt.toSec();
    integral += error * dt.toSec();
    integral = std::min(integral, 1.0);

    double out = p_ * error + i_ * integral + d_ * delta_e;

//    std::cout << "P value:  " << p_ << " D value: " << d_ << " I value: " << i_ << std::endl;

    prev_t = stamp;
    prev_e = error;
    return out;
}

/*
 * Resets the integral part to avoid biases
 */
void PID::resetIntegral(){
    integral = 0.0;
}

/*
 * Sends a boolean true as a topic message if the final goal has been reached
 */
void controller::sendGoalMsg(const bool reached){
    if (!completion_advertised_) {
        completion_advertised_ = true;
        std_msgs::Bool msg;
        msg.data = reached;
        pub_goal_reached_.publish(msg); // Goal reached is being published as msg.data
    }
}

/*
 * Update the parameters from YAML file
 */
void controller::LoadParametersFromYaml() {
    std::string yaml_path = ros::package::getPath("freicar_control_rr") + "/params/" + "ControlParams.yaml";
    YAML::Node base = YAML::LoadFile(yaml_path);
    YAML::Node parameters = base["hyperparameter"];
    L_ = parameters["wheelbase"].as<double>();
    pos_tol_ = parameters["position_tolerance"].as<double>();
    delta_max_ = parameters["steering_angle_limit"].as<double>();
    des_v_ = parameters["desired_velocity"].as<float>();
    throttle_limit_ = parameters["throttle_limit"].as<float>();
    minimum_throttle_limit = parameters["minimum_throttle_limit"].as<float>();
    vmax_ = parameters["vmax"].as<float>();
    curvature_vel_limit_factor = parameters["curvature_vel_limit_factor"].as<float>();
    dist_vel_limit_factor = parameters["distance_vel_limit_factor"].as<float>();
    steering_vel_limit_factor = parameters["steering_vel_limit_factor"].as<float>();
    ld_dist_ = parameters["lookahead_dist"].as<double>();
}

/*
 * Rediscretize a given path so that there is at least "dist" meters between each point
 */
std::vector<tf2::Transform> controller::discretizePath(std::vector<tf2::Transform> &path, float dist){
    std::vector<tf2::Transform> disc_path;
    disc_path.push_back(path.at(0));

//    tf2::Vector3 last_dir;
    for(int i=0; i< (path.size()-1); i++){
        float current_d = dist;
        tf2::Vector3 t1p = path.at(i).getOrigin();
        tf2::Vector3 t2p = path.at(i + 1).getOrigin();
        tf2::Vector3 dir = (t2p - t1p);
        float dir_len = dir.length();

        if(dir_len < 0.05)
            continue;

        while(dir_len > current_d){
            tf2::Transform new_t;
            new_t.setIdentity();
            new_t.setOrigin(current_d * dir.normalize() + t1p);
            disc_path.push_back(new_t);
            current_d += dist;
        }
        disc_path.push_back(path.at(i+1));
    }
    return disc_path;
}


/*
 * Transforms a given path to any target frame
 */
std::vector<tf2::Transform> controller::transformPath(nav_msgs::Path &path, const std::string target_frame){
    std::vector<tf2::Transform> t_path;
    if(path.header.frame_id != target_frame){

        geometry_msgs::TransformStamped tf_msg;
        tf2::Stamped<tf2::Transform> transform;
        tf_msg = tf_buffer_.lookupTransform(path.header.frame_id, target_frame, ros::Time(0));
        tf2::convert(tf_msg, transform);

        for (auto & i : path.poses){
            tf2::Transform t_pose;
            tf2::convert(i.pose, t_pose);
            t_pose = transform * t_pose;
            t_pose.getOrigin().setZ(0);
            t_path.push_back(t_pose);
        }
    }else{
        for (auto & i : path.poses){
            tf2::Transform t_pose;
            tf2::convert(i.pose, t_pose);
            t_pose.getOrigin().setZ(0);
            t_path.push_back(t_pose);
        }
    }
    return t_path;
}

/*
 * Callback function that receives a path
 */
void controller::receivePath(raiscar_msgs::ControllerPath new_path)
{
    // When a new path received, the previous one is simply discarded
    // It is up to the planner/motion manager to make sure that the new
    // path is feasible.
//    ROS_INFO("Received new path");
    pos_error = 0;
//    vel_override_ = new_path.des_vel;
    if (new_path.path_segment.poses.size() > 0)
    {
        /* A small block for taking the path description sent by freicar drive */
        std::vector<float> temp_path_desc;
        for (auto & i : new_path.path_segment.poses){
            temp_path_desc.push_back(i.pose.orientation.w);
        }
        _path_desc = temp_path_desc;

        path_ = transformPath(new_path.path_segment, map_frame_id_);
//        path_ = discretizePath(path_, 0.01); /* Discretizing the path for smoother sontrol */
        goal_reached_ = false;
        /* Change the desired velocity to zero as soon as a new plan comes in
         * This is done to counter the effects of new plan coming in from high level commands.
         * */
        des_v_ = 0;
        completion_advertised_ = false;
        geometry_msgs::Vector3 path_seg;
    }
    else
    {
        std::cout << "No path currently available" << '\n';
        path_ = std::vector<tf2::Transform>();
        goal_reached_ = true;
        completion_advertised_ = true;
        // ROS_WARN_STREAM("Received empty path!");
    }

//    path_ = discretizePath(path_, 0.05);
}

/*
 * Virtual function that needs to be reimplemented
 */
void controller::controller_step(nav_msgs::Odometry odom)
{
    std::cout << "No action implemented ..." << std::endl;
}

void controller::stop_step(std_msgs::Bool stopline_status)
{
    // Getting the pose of the stop line.
    // Getting the pose of the stop line.
    car_stop_status = stopline_status.data;
}

void controller::getRightOfWay(std_msgs::Bool right_of_way_status)
{
    // Getting the pose of the stop line.
    rightOfWay_status = right_of_way_status.data;
}

void controller::getStandingVehicle(std_msgs::Bool Standing_Vehicle_Status)
{
    IsVehicleStanding = Standing_Vehicle_Status.data;
}

void controller::ExtControlCallback(const freicar_common::FreiCarControl::ConstPtr &ctrl_cmd)
{
    std::cout << "ctrl_cmd command:  " << ctrl_cmd->command << ", ctrl_cmd name:  " << ctrl_cmd->name << "\n";
    if(ctrl_cmd->name == "freicar_1"){
        stp_resume_cmd = ctrl_cmd->command;
    }
}

/* Change p, i, d values here. */
controller::controller():pos_tol_(0.1), idx_(0),
                         goal_reached_(true), nh_private_("~"), tf_listener_(tf_buffer_), vel_pid(0.05, 0.15, 0.03){
    // Get parameters from the parameter server
    nh_private_.param<bool>("use_yaml_spawn", use_yaml_spawn_, false);
    if(!use_yaml_spawn_){
        // Load parameters from launch file if given else consider default values
        nh_private_.param<double>("wheelbase", L_, 0.36);
        nh_private_.param<double>("position_tolerance", pos_tol_, 1.5);
        nh_private_.param<double>("steering_angle_limit", delta_max_, 1.22173);
        nh_private_.param<float>("desired_velocity", des_v_, 0.5);
        nh_private_.param<float>("throttle_limit", throttle_limit_, 0.25);
        nh_private_.param<float>("minimum_throttle_limit", minimum_throttle_limit, 0.1);
        nh_private_.param<float>("vmax", vmax_, 5.0);
		nh_private_.param<float>("curvature_vel_limit_factor", curvature_vel_limit_factor, 0.44);
    	nh_private_.param<float>("distance_vel_limit_factor", dist_vel_limit_factor, 0.3);
        nh_private_.param<float>("steering_vel_limit_factor", steering_vel_limit_factor, 2.0);
        nh_private_.param<double>("lookahead_dist", ld_dist_, 0.8);
    }
    // Load parameters for Yaml file
    else
        LoadParametersFromYaml();

    nh_private_.param<std::string>("map_frame_id", map_frame_id_, "map");
    nh_private_.param<std::string>("robot_frame_id", tracker_frame_id, "freicar");
    nh_private_.param<std::string>("target_frame_id", target_frame_id_, tracker_frame_id + "/lookahead");

    front_axis_frame_id_ = tracker_frame_id + "/front_axis";
    rear_axis_frame_id_ = tracker_frame_id + "/rear_axis";

    // Populate messages with static data
    target_p_.header.frame_id = map_frame_id_;
    target_p_.child_frame_id = target_frame_id_;
    target_p_.transform.rotation.w = 1.0;

    car_stop_status = false;
    rightOfWay_status = true;
    IsVehicleStanding = false;
    Request_Overtake_Plan = false;
    stp_resume_cmd = "start";

    // Subscribers to path segment and odometry
    ROS_INFO("Before calling receive path");
    sub_path_ = nh_.subscribe("path_segment", 1, &controller::receivePath, this);

    ROS_INFO("Before calling controller step");
    sub_odom_ = nh_.subscribe("odometry", 1, &controller::controller_step, this);

    ROS_INFO("Before calling stop status step");
    sub_stop_status = nh_.subscribe("stopline_status", 1, &controller::stop_step, this);

    ROS_INFO("Before calling right of way status");
    sub_right_of_way_status = nh_.subscribe("right_of_way", 1, &controller::getRightOfWay, this);

    ROS_INFO("Before calling external control call back function");
    external_control_sub = nh_.subscribe<freicar_common::FreiCarControl>("/freicar_commands",10, &controller::ExtControlCallback, this);

    ROS_INFO("Before calling Overtaking plan");
    Standing_vehicle = nh_.subscribe("Standing_Vehicle", 1, &controller::getStandingVehicle, this);

    // Publishers for the control command and the "reached" message
    pub_acker_ = nh_.advertise<raiscar_msgs::ControlCommand>("control", 1);
    pub_goal_reached_ = nh_.advertise<std_msgs::Bool>("goal_reached", 1);
    pub_overtake_request_ = nh_.advertise<std_msgs::Bool>("freicar_1/request_overtake", 1);

    completion_advertised_ = false;
    current_steering_angle_ = 0;

    projection_t_.header.frame_id = map_frame_id_;
    projection_t_.child_frame_id = target_frame_id_;
    projection_t_.transform.rotation.w = 1.0;


    std::cout << "Pure Pursuit Controller started for frame : " << tracker_frame_id << std::endl;
}