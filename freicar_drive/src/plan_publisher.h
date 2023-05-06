//
// Created by Team Roboracers on 2/13/21.
//

#ifndef FREICAR_DRIVE_PLAN_PUBLISHER_H
#define FREICAR_DRIVE_PLAN_PUBLISHER_H

#include "ros/ros.h"
#include <geometry_msgs/PoseArray.h>
#include "std_msgs/String.h"
#include <freicar_map/planning/lane_star.h>
#include <freicar_map/planning/lane_follower.h>
#include <freicar_map/thrift_map_proxy.h>
#include "map_core/freicar_map.h"
#include "freicar_common/shared/planner_cmd.h"
#include <cstdio>
#include <visualization_msgs/MarkerArray.h>
#include "nav_msgs/Path.h"
#include "std_msgs/Bool.h"
#include <tf2_ros/transform_listener.h>
#include <tf2/transform_datatypes.h>
#include <tf2/utils.h>
#include <tf2/convert.h>

#include "freicar_common/FreiCarControl.h"
#include <cstdio>
#include "nav_msgs/Path.h"
#include <visualization_msgs/MarkerArray.h>
#include "raiscar_msgs/ControllerPath.h"
#include "image_boundingboxinfo_publisher/boxes.h"
#include "image_boundingboxinfo_publisher/box.h"
#include "tf2/transform_datatypes.h"
#include "tf2/transform_storage.h"
#include <tf2/convert.h>
#include <tf2/utils.h>
#include "Eigen/Dense"


class plan_publisher {
public:
    plan_publisher(std::shared_ptr<ros::NodeHandle> n);
    std::shared_ptr<ros::NodeHandle> n_;

    void GetParticles(const geometry_msgs::PoseArray msg);
    void GoalReachedStatus(const std_msgs::Bool reached);
    void RequestOvertakeStatus(const std_msgs::Bool Overtake_Request);
    void DepthInfoStatus(const std_msgs::Bool car_closeby);
    void ExtControlCallback(const freicar_common::FreiCarControl::ConstPtr &ctrl_cmd);
    void CommandNewPlanner(bool goal_reach_flg, bool cmd_changed_flag);
    void GetRightOfWay(std::vector<Eigen::Vector3f> ObservedCars);
    void BoundingBoxCallback(const image_boundingboxinfo_publisher::boxesPtr &msg);
    bool DetectRightOfWaySign(freicar::map::Map &map, const std::string &current_lane_uuid, bool b);

    std::vector<freicar::mapobjects::Point3D>OvertakePlan();

    ros::Subscriber sub;
    ros::Subscriber goal_reached,depth_info;
    ros::Subscriber external_control_sub;
    ros::Subscriber boundingbox_sub;
    ros::Subscriber request_overtake;

    ros::Publisher path_to_follow;
    ros::Publisher tf,overtake_plan;
    ros::Publisher stopline_status;
    ros::Publisher right_of_way_status;
    ros::Publisher Overtake_status;

    freicar::mapobjects::Point3D start_point ;
    std_msgs::Bool car_stop_status,standing_vehicle;
    std::string old_lane_uuid;

    float start_angle = 0.0;
    bool car_depth_ = false;
    bool Send_Overtake_plan = false;
    ros::Time time_when_last_stopped;
    ros::Time last_time_of_no_right_of_way;
    tf2_ros::Buffer tf_buffer_;

private:
    freicar::enums::PlannerCommand command = freicar::enums::PlannerCommand::STRAIGHT;
    std::string planner_cmd;
    bool goal_reached_flag;
    bool command_changed = false;
    std_msgs::Bool right_of_way;

    tf2_ros::TransformListener tf_obs_agent_listener;

    int findPathDescription(freicar::mapobjects::Lane::Connection description);
};

#endif //FREICAR_DRIVE_PLAN_PUBLISHER_H
