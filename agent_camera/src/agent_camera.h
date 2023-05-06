//
// Created and implemented by Venkat as part of Team Roboracers on 3/19/21.
//

#ifndef AGENT_CAMERA_AGENT_CAMERA_NODE_H
#define AGENT_CAMERA_AGENT_CAMERA_NODE_H

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "rospy_tutorials/Floats.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3.h>
#include "std_msgs/Bool.h"
#include <tf2_ros/transform_broadcaster.h>
#include "image_boundingboxinfo_publisher/box.h"
#include "image_boundingboxinfo_publisher/boxes.h"


class agent_camera {
public:
    agent_camera(std::shared_ptr<ros::NodeHandle> n);

    void RGBCallback(const sensor_msgs::ImageConstPtr& msg);
    void DepthCallback(const sensor_msgs::ImageConstPtr& msg);
    void BoundingBox(const image_boundingboxinfo_publisher::boxesPtr &msg);

    std::shared_ptr<ros::NodeHandle> n_;
    ros::Subscriber rgb_sub;
    ros::Subscriber depth_sub;
    ros::Subscriber bb_sub;
    ros::Publisher car_ahead_pub_;

private:
    double fx=(725.3607788085938/2), fy=(725.3607788085938/2), cx=(596.6277465820312 * .5333333333), cy=(337.6268615722656/2); //intrinsic parameters
    float car_ahead_threshold=0.5;  //try other values rangind between 0.25 and 0.45
    geometry_msgs::TransformStamped camera_to_map_proj_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    image_boundingboxinfo_publisher::boxes bounding_boxes_;
    image_boundingboxinfo_publisher::box bounding_box;

    Eigen::Matrix3d intrinsic_mat, intrinsic_mat_inv;
};


#endif //AGENT_CAMERA_AGENT_CAMERA_NODE_H
