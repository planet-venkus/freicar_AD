/*
 * Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
 * Project: FreiCAR
 * Do NOT distribute this code to anyone outside the FreiCAR project
 */

/*
 * This file is the main file of the particle filter. It sets up ROS publisher/subscriber, defines the callbacks and
 * creates the particle filter class
 */

#include "ros/ros.h"
#include "ros_pf_localization.h"
#include "visualization_msgs/MarkerArray.h"
#include <yaml-cpp/yaml.h>
#include <sstream>
#include "std_msgs/String.h"

// The freicar map class is a singleton, get the instance of that singleton here
freicar::map::Map &Localizer::map_ = freicar::map::Map::GetInstance();

/*
 * Fetching spawn location values from YAML file
 */
void Localizer::LoadSpawnPoseFromYaml(std::string map_name) {
    std::string yaml_path = ros::package::getPath("freicar_launch") + "/spawn_positions/" + map_name + "_spawn.yaml";
    YAML::Node base = YAML::LoadFile(yaml_path);
    YAML::Node pose_node = base["spawn_pose"];
    init_x = pose_node["x"].as<float>();
    init_y = pose_node["y"].as<float>();
    init_z = pose_node["z"].as<float>();
    init_heading = pose_node["heading"].as<float>();
}

/*
 * Entry class of ROS node. It does the following:
 * 1. Sets up all subscribers and publishers
 * 2. Initializes Map
 * 3. Initializes all particles of the particle filter class
 */
Localizer::Localizer(std::shared_ptr<ros::NodeHandle> n) : n_(n), it_(*n) {
    ros::Duration sleep_time(1);
//    odo_sub_ = n_->subscribe("/freicar_1/odometry_noise", 1, &Localizer::OdoCallback, this);
    odo_sub_ = n_->subscribe("/freicar_1/odometry", 1, &Localizer::OdoCallback, this);
    marker_sub_ = n_->subscribe("traffic_signs", 1, &Localizer::markerCallback, this);
    last_odo_update_ = ros::Time::now();

    // Get the initial pose of the car
    nh_private_.param<bool>("use_yaml_spawn", use_yaml_spawn_, false);
    if(!use_yaml_spawn_){
        nh_private_.param<float>("init_x", init_x, 0.0);
        nh_private_.param<float>("init_y", init_y, 0.0);
        nh_private_.param<float>("init_z", init_z, 0.0);
        nh_private_.param<float>("heading", init_heading, 0.0);
    }
    else{
        nh_private_.param<std::string>("map_name", map_name, "freicar_1");
        LoadSpawnPoseFromYaml(map_name);
    }

    freicar::map::ThriftMapProxy map_proxy("127.0.0.1", 9091, 9090);
    std::string map_path;
    if (!ros::param::get("/map_path", map_path)) {
        ROS_ERROR("could not find parameter: map_path! map initialization failed.");
        return;
    }

    if (!ros::param::get("~use_lane_regression", use_lane_reg_)) {
        ROS_INFO("could not find parameter: use_lane_regression! default: do not use lane_regression.");
        use_lane_reg_ = false;
    }

    if (!ros::param::get("~evaluate", evaluate_)) {
        ROS_INFO("could not find parameter: evaluate! default: do not evaluate.");
        use_lane_reg_ = false;
    }


    if (use_lane_reg_) {
        image_sub_ = it_.subscribe("/freicar_1/sim/camera/rgb/front/reg_bev", 1, &Localizer::RegCallback, this);
    }
// if the map can't be loaded
    if (!map_proxy.LoadMapFromFile(map_path)) {
        ROS_INFO("could not find thriftmap file: %s, starting map server...", map_path.c_str());
        map_proxy.StartMapServer();
        // stalling main thread until map is received
        while (freicar::map::Map::GetInstance().status() == freicar::map::MapStatus::UNINITIALIZED) {
            ROS_INFO("waiting for map...");
            ros::Duration(1.0).sleep();
        }
        ROS_INFO("map received!");
        // Thrift creates a corrupt file on failed reads, removing it
        remove(map_path.c_str());
        map_proxy.WriteMapToFile(map_path);
        ROS_INFO("saved new map");
    }
    ros::Duration(2.0).sleep();
    freicar::map::Map::GetInstance().PostProcess(0.22);  // densifies the lane points

    visualizer_ = std::make_shared<ros_vis>(n_);
    p_filter = std::make_shared<particle_filter>(&map_, visualizer_, use_lane_reg_, init_x, init_y, init_z, init_heading);

    sleep_time.sleep();
//    visualizer_->SendPoints(p_filter->getMapKDPoints(), "map_points", "/map");
    //p_filter->InitParticlesAroundPose(GetTf());
    ROS_INFO("Sent map points...");
}

/*
 * Receives /base_link relative to /map as Eigen transformation matrix.
 * WARNING, IMPORTANT: This is ground truth, so don't use it anywhere in the particle filter. This is just for reference
 */
Eigen::Transform<float, 3, Eigen::Affine> Localizer::GetTf(ros::Time time) {
    Eigen::Transform<float, 3, Eigen::Affine> gt_t = Eigen::Transform<float, 3, Eigen::Affine>::Identity();
    tf::StampedTransform transform;
    try {
        listener.lookupTransform("/map", "/freicar_1/base_link",
                                 time, transform);
    }
    catch (tf::TransformException ex) {
        ROS_ERROR("%s", ex.what());
        ros::Duration(1.0).sleep();
    }

    gt_t.translate(Eigen::Vector3f(transform.getOrigin().x(), transform.getOrigin().y(), 0.));
    gt_t.rotate(Eigen::Quaternionf(transform.getRotation().getW(), transform.getRotation().getX(),
                                   transform.getRotation().getY(), transform.getRotation().getZ()));
    return gt_t;
}

// Destructor
Localizer::~Localizer(){
    std::cout << "Shutting down localizer..." << std::endl;
    if (evaluate_){
        double mean_error = aggregated_error_ / num_measurements_;
        std::cout << "Average error during this run [m]: " << mean_error << std::endl;
    }
}

// Callback function for getting the FreiCAR Sign detections
void Localizer::markerCallback(const freicar_common::FreiCarSignsConstPtr &markers_msg) {
    last_sign_msg_ = *markers_msg;
    std::vector<Sign> sign_observations;

    for (size_t i = 0; i < last_sign_msg_.signs.size(); i++) {
        freicar_common::FreiCarSign s_msg = last_sign_msg_.signs.at(i);
        unsigned int in_id = s_msg.type;
        // check valid range of ids (AudiCup 18)
        if (in_id >= 0 && in_id <= 17) {
            Sign new_sign;
            new_sign.id = s_msg.type;
            new_sign.position = Eigen::Vector3f(s_msg.x, s_msg.y, s_msg.z);
            // Convert id into type string
            new_sign.type = SignIdToTypeString(new_sign.id);
            sign_observations.push_back(new_sign);
        }
    }
    latest_signs_ = std::pair<std::vector<Sign>,ros::Time>(sign_observations, markers_msg->header.stamp);
}


/*
 * Callback function for getting the odometry in order to process the motion model later.
 * This function also visualizes the particles. You have the option to either just show the best particle or
 * showing a mean particle consisting out of the k best particles
 */
void Localizer::OdoCallback(const nav_msgs::OdometryConstPtr &msg) {
    // Apply motion model to all particles
    geometry_msgs::TransformStamped trans_pose;
    nav_msgs::Odometry conv_odometry;
    p_filter->MotionStep(*msg);


    //visualizer_->SendBestParticle(p_filter->getBestParticle(), "/map");
    Particle best_particle = p_filter->getMeanParticle(3000);
    visualizer_->SendBestParticle(best_particle, "map");
    last_odo_update_ = msg->header.stamp;

    trans_pose.header.frame_id = "map";
    trans_pose.child_frame_id = "freicar_1/handle";
    trans_pose.header.stamp = last_odo_update_;
    trans_pose.transform.translation.x = best_particle.transform.translation().x();
    trans_pose.transform.translation.y = best_particle.transform.translation().y();
    trans_pose.transform.translation.z = best_particle.transform.translation().z();
    Eigen::Quaternionf q(best_particle.transform.rotation().matrix()) ;
    trans_pose.transform.rotation.w = q.w();
    trans_pose.transform.rotation.x = q.x();
    trans_pose.transform.rotation.y = q.y();
    trans_pose.transform.rotation.z = q.z();

//    std::cout << "pose_msg.orientation.w: " << pose_msg.orientation.w << ", pose_msg.orientation.x: " << pose_msg.orientation.x << ", pose_msg.orientation.y: " << pose_msg.orientation.y << ", pose_msg.orientation.z:" << pose_msg.orientation.z;
    tf_broadcaster.sendTransform(trans_pose);

    if (evaluate_ && first_observation_received_) {
        Eigen::Transform<float, 3, Eigen::Affine> gt_pose = this->GetTf(ros::Time(0));
        Eigen::Vector3f particle_pos = Eigen::Vector3f(best_particle.transform.translation().x(), best_particle.transform.translation().y(), 0.);
        float position_error = std::sqrt((gt_pose.translation() - particle_pos).squaredNorm());
//        poerr.push_back(position_error);
        aggregated_error_ += position_error;
        num_measurements_++;
//        std::cout << "Position error [m]: " << position_error << std::endl;
    }
}

/*
 * Converts the sign id to the corresponding type string
 * E.g The sign with id 0 belongs to the type string "CrossingRight"
 */
std::string Localizer::SignIdToTypeString(int id) {
    switch (id) {
        case 0:
            return "CrossingRight";
        case 1:
            return "Stop";
        case 2:
            return "Parking";
        case 3:
            return "RightOfWay";
        case 4:
            return "Straight";
        case 5:
            return "GiveWay";
        case 6:
            return "PedestrianCrossing";
        case 7:
            return "Roundabout";
        case 8:
            return "NoOvertaking";
        case 9:
            return "NoEntry";
        case 10:
            return "Position";
        case 11:
            return "OneWay";
        case 12:
            return "Roadworks";
        case 13:
            return "Speed50";
        case 14:
            return "Speed100";
        case 15:
            return "Spare1";
        case 16:
            return "Position";
        case 17:
            return "Position";
        default:
            return "Position";
    }
    return "";
}

/*
 * This function collects all latest observation messages and passes it to the sensor model
 */
void Localizer::StartObservationStep() {
    // Got observation -> weight particles
    if (sensor_model_thread_.valid()) {
        if (sensor_model_thread_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
//            std::cout << "Sensor model is busy, waiting..." << std::endl;
            return;
        } else {
            //std::cout << "Sensor model is ready..." << std::endl;
        }
    }

    std::vector<Sign> observation_signs;
    std::vector<cv::Mat> observation_reg;

    if (!latest_signs_.first.empty()) {
        if (abs((latest_signs_.second - last_odo_update_).toSec()) < 0.05) {
            observation_signs = (latest_signs_.first);
            first_observation_received_ = true;
        }
        latest_signs_.first.clear();
    }

    if (use_lane_reg_) {
        if (!latest_lane_reg_.first.empty()) {
            if (abs((latest_lane_reg_.second - last_odo_update_).toSec()) < 0.05) {
                observation_reg = (latest_lane_reg_.first);
                first_observation_received_ = true;
            }

            latest_lane_reg_.first.clear();
        }
    }
    if (!observation_reg.empty() || !observation_signs.empty()) {
        sensor_model_thread_ = std::async(std::launch::async, &particle_filter::ObservationStep, p_filter,
                                          observation_reg,
                                          observation_signs);
    }

//    ////ROS TO BE REMOVED ///////////////////////////////////////////////////////////////////
//    int sample_size = 1000;
//    cv::Mat lreg;
//    std::vector<Eigen::Vector3f> sampled_lreg;
//    std::vector<int> sampled_idxs;
//    for (int i = 0; i < observation_reg.size(); i++) {
//        cv::findNonZero(observation_reg.at(i), lreg);
//    }
////        std::cout<<"lreg size = "<<lane_regression.at(0).type()<<'\n';
////        std::cout<<"lreg_nonzero size = "<<lreg.size()<<'\n';
////        std::cout<<"lreg_nonzero at 0,1 = "<<lreg.at<int>(0, 1)<<'\n';
////        std::cout<<"lreg_nonzero at 1,1 = "<<lreg.at<int>(1, 1)<<'\n';
////        std::cout<<"lreg_nonzero at 2,0 = "<<lreg.at<int>(2, 0)<<'\n';
////        std::cout<<"lreg_nonzero at height = "<<lreg.size().height<<'\n';
////
////        std::cout<<"lreg_nonzero  : "<<lreg<<'\n';
//    std::default_random_engine sample_gen;
////        std::discrete_distribution<int> distribution lreg;
//    for (int i = 0; i < sample_size; i++) {
//        int n = lreg.size().height - 1;
//        int sampled_idx = rand() % n;
////           std::cout<<"Sampled idx = "<<sampled_idx<<'\n';
//        sampled_lreg.push_back(
//                Eigen::Vector3f(lreg.at<int>(sampled_idx, 0) / 100, lreg.at<int>(sampled_idx, 1) / 100, 0));
////        // Write code for lane regression here.
////        // Use getNearestPoints function
////        // Sample points
//    }
//
//    std::cout << "Reached the NODE inside execution part" << std::endl;
//    PointCloud<float> pc;
//
//    visualization_msgs::MarkerArray ma;
//    for(int i=0; i<sampled_lreg.size(); i++)
//    {
//        Point_KD<float> sp;
//        sp.x = sampled_lreg[i].x();
//        sp.y =  sampled_lreg[i].y();
//        pc.pts.push_back(sp);
//        visualization_msgs::Marker marker;
//        marker.pose.position.x = sampled_lreg[i].x();
//        marker.pose.position.y = sampled_lreg[i].y();
//        marker.pose.position.z = sampled_lreg[i].z();
//        marker.scale.x = 0.01;
//        marker.scale.y = 0.01;
//        marker.scale.z = 0.01;
//        marker.id = i;
//        marker.color.g = 1.0;
//        marker.color.r = 0.0;
//        marker.color.b = 0.0;
//        marker.color.a = 1.0;
//        marker.header.frame_id = "freicar_1/base_link";
//        marker.header.stamp = ros::Time::now();
//        ma.markers.push_back(marker);
//    }
//
//    visualizer_->SendPoints(pc, "Sampled points", "freicar_1/base_link", 0.0, 1.0, 0.0);
//        visualizer_->SendSample(ma);
//
////        marker_pub.publish(ma);
//
//    }
}

/*
 * Lane Regression callback (BEV view). If the lane regression is provided it will go into the sensor-model and
 * improves the overall performance
 */
void Localizer::RegCallback(const sensor_msgs::ImageConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr;
    static const std::string OPENCV_WINDOW = "Img window";
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat reg = cv_ptr->image;
//    cv::namedWindow(OPENCV_WINDOW);
//    cv::imshow(OPENCV_WINDOW, reg);
//    cv::waitKey(3);
    std::vector<cv::Mat> mat_vec;
    mat_vec.push_back(reg);
    latest_lane_reg_ = std::pair<std::vector<cv::Mat>, ros::Time>(mat_vec, msg->header.stamp);}

// Main function that inits ROS, creates the localizer class and spins the system
int main(int argc, char **argv) {
    ros::init(argc, argv, "freicar_particle_filter");

    std::shared_ptr<ros::NodeHandle> n = std::make_shared<ros::NodeHandle>();

    ros::Rate loop_rate(100);

    std::cout << "Particle filter localization node started: " << std::endl;

    Localizer loc(n);

    while (ros::ok()) {
        ros::spinOnce();
        loc.StartObservationStep();
        loop_rate.sleep();
    }

    return 0;
}
