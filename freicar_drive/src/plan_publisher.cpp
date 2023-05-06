//
// Created by Team Roboracers on 2/13/21.
//
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/PoseArray.h"
#include <freicar_map/planning/lane_star.h>
#include <freicar_map/planning/lane_follower.h>
#include <freicar_map/thrift_map_proxy.h>
#include "map_core/freicar_map.h"
#include "freicar_common/shared/planner_cmd.h"
#include <cstdio>
#include "nav_msgs/Path.h"
#include <visualization_msgs/MarkerArray.h>
#include "plan_publisher.h"
#include "raiscar_msgs/ControllerPath.h"

// Getting best partcle of the car
//std::shared_ptr<ros::NodeHandle> node_handle = std::make_shared<ros::NodeHandle>();
plan_publisher::plan_publisher(std::shared_ptr<ros::NodeHandle> n): n_(n), tf_obs_agent_listener(tf_buffer_)
{
    sub = n_->subscribe("best_particle", 10, &plan_publisher::GetParticles, this);
    goal_reached = n_->subscribe("/freicar_1/goal_reached", 1, &plan_publisher::GoalReachedStatus, this);
    request_overtake = n_->subscribe("freicar_1/request_overtake",1, &plan_publisher::RequestOvertakeStatus,this);
    depth_info = n_->subscribe("/car_ahead", 1, &plan_publisher::DepthInfoStatus, this);

    external_control_sub = n_->subscribe("/freicar_commands",10, &plan_publisher::ExtControlCallback, this);
	boundingbox_sub = n_->subscribe("/bbsarray", 1, &plan_publisher::BoundingBoxCallback, this);
    path_to_follow = n_->advertise<raiscar_msgs::ControllerPath>("/freicar_1/path_segment", 10);
    stopline_status = n_->advertise<std_msgs::Bool>("stopline_status", 1);
    right_of_way_status = n_->advertise<std_msgs::Bool>("right_of_way", 1);
    Overtake_status = n_->advertise<std_msgs::Bool>("Standing_Vehicle", 1);

    tf = n_->advertise<visualization_msgs::MarkerArray>("planner_debug", 10, true);
	right_of_way.data = true;
    overtake_plan = n_->advertise<visualization_msgs::MarkerArray>("overtake_planner", 10, true);

}

void PublishPlan (freicar::planning::Plan& plan, double r, double g, double b, int id, const std::string& name, ros::Publisher& pub) {
    visualization_msgs::MarkerArray list;
    visualization_msgs::Marker *step_number = new visualization_msgs::Marker[plan.size()];
    int num_count = 0;
    visualization_msgs::Marker plan_points;
    plan_points.id = id;
    plan_points.ns = name;
    plan_points.header.stamp = ros::Time();
    plan_points.header.frame_id = "map";
    plan_points.action = visualization_msgs::Marker::ADD;
    plan_points.type = visualization_msgs::Marker::POINTS;
    plan_points.scale.x = 0.03;
    plan_points.scale.y = 0.03;
    plan_points.pose.orientation = geometry_msgs::Quaternion();
    plan_points.color.b = b;
    plan_points.color.a = 0.7;
    plan_points.color.g = g;
    plan_points.color.r = r;
    geometry_msgs::Point p;
    for (size_t i = 0; i < plan.size(); ++i) {
        step_number[i].id = ++num_count + id;
        step_number[i].pose.position.x = p.x = plan[i].position.x();
        step_number[i].pose.position.y = p.y = plan[i].position.y();
        p.z = plan[i].position.z();
        step_number[i].pose.position.z = plan[i].position.z() + 0.1;
        step_number[i].pose.orientation = geometry_msgs::Quaternion();
        step_number[i].ns = name + "_nums";
        step_number[i].header.stamp = ros::Time();
        step_number[i].header.frame_id = "map";
        step_number[i].action = visualization_msgs::Marker::ADD;
        step_number[i].type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        step_number[i].text = std::to_string(i);
        step_number[i].scale.z = 0.055;
        step_number[i].color = plan_points.color;
        list.markers.emplace_back(step_number[i]);
        plan_points.points.emplace_back(p);
    }
    list.markers.emplace_back(plan_points);
    pub.publish(list);
    delete[] step_number;
}

void Overtake_PlanPublish (std::vector<freicar::mapobjects::Point3D> plan3dps, double r, double g, double b, int id, const std::string& name, ros::Publisher& pub) {
    visualization_msgs::MarkerArray list;
    visualization_msgs::Marker *step_number = new visualization_msgs::Marker[plan3dps.size()];
    int num_count = 0;
    visualization_msgs::Marker plan_points;
    plan_points.id = id;
    plan_points.ns = name;
    plan_points.header.stamp = ros::Time();
    plan_points.header.frame_id = "map";
    plan_points.action = visualization_msgs::Marker::ADD;
    plan_points.type = visualization_msgs::Marker::POINTS;
    plan_points.scale.x = 0.03;
    plan_points.scale.y = 0.03;
    plan_points.pose.orientation = geometry_msgs::Quaternion();
    plan_points.color.b = b;
    plan_points.color.a = 0.7;
    plan_points.color.g = g;
    plan_points.color.r = r;
    geometry_msgs::Point p;
    for (size_t i = 0; i < plan3dps.size(); ++i) {
        step_number[i].id = ++num_count + id;
        step_number[i].pose.position.x = p.x = plan3dps[i].x();
        step_number[i].pose.position.y = p.y = plan3dps[i].y();
        p.z = plan3dps[i].z();
        step_number[i].pose.position.z = plan3dps[i].z() + 0.1;
        step_number[i].pose.orientation = geometry_msgs::Quaternion();
        step_number[i].ns = name + "_nums";
        step_number[i].header.stamp = ros::Time();
        step_number[i].header.frame_id = "map";
        step_number[i].action = visualization_msgs::Marker::ADD;
        step_number[i].type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        step_number[i].text = std::to_string(i);
        step_number[i].scale.z = 0.055;
        step_number[i].color = plan_points.color;
        list.markers.emplace_back(step_number[i]);
        plan_points.points.emplace_back(p);
    }
    list.markers.emplace_back(plan_points);
    pub.publish(list);
    delete[] step_number;
}

// Call back function to check the availability of right of way.
void plan_publisher::GetRightOfWay(std::vector<Eigen::Vector3f> ObservedCars){

    // Fetch the upcoming junction of the car's current location
    auto &map = freicar::map::Map::GetInstance();
    auto p_closest = map.FindClosestLanePoints(start_point.x(), start_point.y(), start_point.z(), 1)[0].first;
    auto current_lane_uuid = p_closest.GetLaneUuid();
    bool rightOfWaySignDetected = false;
    rightOfWaySignDetected = DetectRightOfWaySign(map, current_lane_uuid, rightOfWaySignDetected);
    auto upcoming_junction = map.GetUpcomingJunctionID(p_closest.GetLaneUuid());

    // Check for RightOfWay if we are approaching a junction
    if(upcoming_junction != -1){
        // Scan for any observed agents who could claim RightOfWay
        for(int i = 0; i < ObservedCars.size(); i++){
            auto obs_p_closest = map.FindClosestLanePoints(ObservedCars[i].x(), ObservedCars[i].y(), ObservedCars[i].z(), 1)[0].first;
            // Observered car's current lane
            auto obs_current_lane = map.FindLaneByUuid(obs_p_closest.GetLaneUuid());
            //If obs car's lane is not a Junction
            if(!obs_current_lane->IsJunctionLane() ){
                int upcoming_junction_id_of_obs_car = map.GetUpcomingJunctionID(obs_p_closest.GetLaneUuid());
                //if obs car has no upcoming junction or obs car's upcoming junction id is different than ours
                if(upcoming_junction_id_of_obs_car == -1 || upcoming_junction_id_of_obs_car != upcoming_junction){
//As some of the lane sections around a junction are short therefore checking if the upcoming lane of the observed car has an upcoming junction
//                    auto obs_next_lane = obs_current_lane->GetConnection(freicar::mapobjects::Lane::STRAIGHT);
//                    if(map.GetUpcomingJunctionID(obs_next_lane->GetUuid().GetUuidValue()) == -1){
//                        right_of_way.data = true;
//                    }
//                    else{
//                        right_of_way.data = false;
//                        break;
//                    }
                    //if we detect the right of way sign
                    if(rightOfWaySignDetected){
                        right_of_way.data = true;
                    }else if(!rightOfWaySignDetected){
                        right_of_way.data = false;
                        break;
                    }
                }
                else if(upcoming_junction_id_of_obs_car != -1 && upcoming_junction_id_of_obs_car == upcoming_junction){
                    right_of_way.data = false;
                    break;
                }
            }
            else if(obs_current_lane->IsJunctionLane()){
                right_of_way.data = false;
                break;
            }
        }
    }
    else if(upcoming_junction == -1){
        // We have the RightOfWay as we are not approaching any junction.
        right_of_way.data = true;
    }
}

bool plan_publisher::DetectRightOfWaySign(freicar::map::Map &map, const std::string &current_lane_uuid, bool rightOfWaySignDetected) {
    auto current_lane = map.FindLaneByUuid(current_lane_uuid);
    std::vector<const freicar::mapobjects::Roadsign*> car_lane_signs;
    car_lane_signs = current_lane->GetRoadSigns();
    for (auto lane_sign : car_lane_signs) {
        if (lane_sign->GetSignType() == "RightOfWay")
            rightOfWaySignDetected = true;
    }
    return rightOfWaySignDetected;
}

// Call back function to process higher commands.
void plan_publisher::ExtControlCallback(const freicar_common::FreiCarControl::ConstPtr &ctrl_cmd)
{
    if(ctrl_cmd->name == "freicar_1"){
        goal_reached_flag = false;
        planner_cmd = ctrl_cmd->command;
        if(planner_cmd == "right")
        {
            command = freicar::enums::PlannerCommand::RIGHT;
            command_changed = true;
        }
        else if(planner_cmd == "left")
        {
            command = freicar::enums::PlannerCommand::LEFT;
            command_changed = true;
        }
        else if(planner_cmd == "right")
        {
            command = freicar::enums::PlannerCommand::RIGHT;
            command_changed = true;
        }
        else if(planner_cmd == "straight")
        {
            command = freicar::enums::PlannerCommand::STRAIGHT;
            command_changed = true;
        }

        if(!goal_reached_flag) {
            CommandNewPlanner(goal_reached_flag, command_changed);
        }
    }
    else{
        command_changed = false; //command changes flag reset
    }

}

// Call back function to publish overtake plan.
void plan_publisher::RequestOvertakeStatus(const std_msgs::Bool Overtake_Request) {
    if(Overtake_Request.data && car_depth_){
        geometry_msgs::PoseStamped pose_msg;
        raiscar_msgs::ControllerPath rais_control_msg;

//        ros::Time end_time = ros::Time::now();
//        ros::Duration duration = end_time - overtake_time_start;

        auto overtake_points = plan_publisher::OvertakePlan();
        Overtake_PlanPublish(overtake_points, 0.0, 0.0, 1.0, 301, "plan_1", overtake_plan);


        // To publish on ROS
        rais_control_msg.path_segment.header.stamp = ros::Time::now();
        rais_control_msg.path_segment.header.frame_id = "map";
        rais_control_msg.path_segment.header.seq = 0;

        // To publish on ROS
        for (size_t i = 0; i < overtake_points.size(); ++i) {
            pose_msg.pose.position.x = overtake_points[i].x();
            pose_msg.pose.position.y = overtake_points[i].y();
            pose_msg.pose.position.z = overtake_points[i].z();

            rais_control_msg.path_segment.poses.push_back(pose_msg);
        }
        path_to_follow.publish(rais_control_msg);
    }
    else if(!car_depth_){
        CommandNewPlanner(true, command_changed);
    }
}

// Call back function to fetch new plan.
void plan_publisher::GoalReachedStatus(const std_msgs::Bool reached) {
    goal_reached_flag = reached.data;
    if(!command_changed){
        CommandNewPlanner(goal_reached_flag, command_changed);
    }
    goal_reached_flag = false; //reset the flag
}

void plan_publisher::DepthInfoStatus(const std_msgs::Bool car_closeby) {
    if(car_closeby.data){
        car_depth_ = car_closeby.data;
    }
    else{
        car_depth_ = false;
    }
}

void plan_publisher::BoundingBoxCallback(const image_boundingboxinfo_publisher::boxesPtr &msg) {
    std::vector<Eigen::Vector3f> observed_car_position_s;
    image_boundingboxinfo_publisher::boxes bounding_boxes_ = *msg;
    Eigen::Vector3f observed_car_position_;

    for (int i = 0; i < bounding_boxes_.bounding_boxes.size(); ++i) {
        geometry_msgs::TransformStamped tf_msg;
        tf2::Stamped<tf2::Transform> cam_t_map;
        try
        {
            tf_msg = tf_buffer_.lookupTransform("map", "freicar_1/depth_point_" + std::to_string(i), ros::Time::now(), ros::Duration(3));
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN_STREAM(ex.what());
        }
        tf2::convert(tf_msg, cam_t_map);
        observed_car_position_.x() = cam_t_map.getOrigin().x(); // offset 0.3
        observed_car_position_.y() = cam_t_map.getOrigin().y(); // offset 0.15
        observed_car_position_.z() = cam_t_map.getOrigin().z();
        observed_car_position_s.push_back(observed_car_position_);
    }

    // Function call to check and publish right of way only if we have any cars observed.
    if(observed_car_position_s.size() > 0) {
        GetRightOfWay(observed_car_position_s);
    }
    else {
        right_of_way.data = true;
    }

    const freicar::mapobjects::Lane *Observed_vehicle_lane;
    const freicar::mapobjects::Lane *current_lane;
    const freicar::mapobjects::Lane *overtake_current_lane;

    auto &map = freicar::map::Map::GetInstance();

    // Fetch the stop line to stop at if we do not have right of way.
    if(!right_of_way.data){
        // To Fetch the current lane.
        auto p_closest = map.FindClosestLanePoints(start_point.x(), start_point.y(), start_point.z(), 1)[0].first;
        auto current_lane_uuid = p_closest.GetLaneUuid();
        current_lane = map.FindLaneByUuid(current_lane_uuid);

        // Fetch the stop line to stop
        freicar::mapobjects::Point3D stoplinepos;
        float distance;
        const freicar::mapobjects::Stopline *stopline = current_lane->GetStopLine();
        stoplinepos = stopline->GetPosition();
        distance = stoplinepos.ComputeDistance(start_point);
        ros::Time now = ros::Time::now();
        if(distance < 0.7 && ((now-last_time_of_no_right_of_way).toSec() > 20)){
            // Publish the RightOfWay stop status
            right_of_way_status.publish(right_of_way);
            last_time_of_no_right_of_way = ros::Time::now();
        }
    }
	else{
        // Keep Publishing that we have RightOfWay
        right_of_way_status.publish(right_of_way);
    }
	
	

    if(observed_car_position_s.size() >= 1){
        auto p_closest_overtake = map.FindClosestLanePoints(observed_car_position_.x(), observed_car_position_.y(), observed_car_position_.z(), 3)[0].first;
        Observed_vehicle_lane = map.FindLaneByUuid(p_closest_overtake.GetLaneUuid());

        auto my_closest = map.FindClosestLanePoints(start_point.x(), start_point.y(), start_point.z(), 3)[0].first;
        overtake_current_lane = map.FindLaneByUuid(my_closest.GetLaneUuid());

        std_msgs::Bool overtake_status_msg;
        if((Observed_vehicle_lane->GetLaneDirection() == overtake_current_lane->GetLaneDirection())&&(!Observed_vehicle_lane->IsJunctionLane())&&(car_depth_)){
//        ros::Time overtake_time_start = ros::Time::now();
//        ros::Duration duration = end_time - begin_time;
            overtake_status_msg.data = true;
            Overtake_status.publish(overtake_status_msg);
        }
        else{
            overtake_status_msg.data = false;
            Overtake_status.publish(overtake_status_msg);
        }
    }
	
}

// Call back function to get the current location of the car.
void plan_publisher::GetParticles(const geometry_msgs::PoseArray msg){
    // Fetching the current location of the car which is plan's starting point
    freicar::mapobjects::Point3D current_point(msg.poses.data()->position.x, msg.poses.data()->position.y, msg.poses.data()->position.z);
    start_point = current_point;
    double x_ang = msg.poses.data()->orientation.x;
    double y_ang = msg.poses.data()->orientation.y;
    double z_ang = msg.poses.data()->orientation.z;
    double w_ang = msg.poses.data()->orientation.w;
    start_angle = atan2(2.0f * (w_ang * z_ang + x_ang * y_ang), 1.0f - 2.0f * (y_ang * y_ang + z_ang * z_ang));

    // To Fetch the current lane.
    const freicar::mapobjects::Lane *current_lane;
    auto &map = freicar::map::Map::GetInstance();
    auto p_closest = map.FindClosestLanePoints(current_point.x(), current_point.y(), current_point.z(), 1)[0].first;
    auto current_lane_uuid = p_closest.GetLaneUuid();
    current_lane = map.FindLaneByUuid(current_lane_uuid);

    // To Fetch the sign boards connected to the current lane.
    freicar::mapobjects::Point3D stoplinepos;
    float distance;
    // If stop sign detected.
    if(current_lane->HasRoadSign("Stop")) {
        const freicar::mapobjects::Stopline *stopline = current_lane->GetStopLine();
        stoplinepos = stopline->GetPosition();
        distance = stoplinepos.ComputeDistance(current_point);
        if(distance < 0.7){
            car_stop_status.data = true;
        }
    }
    // No stop sign detected.
    else {
        car_stop_status.data = false;
    }
    ros::Time now = ros::Time::now();
    // If car needed to be stopped and for a different lane than before.
    if(car_stop_status.data == true) {
        if (current_lane_uuid != old_lane_uuid) {
            old_lane_uuid = current_lane_uuid;
            stopline_status.publish(car_stop_status);
            time_when_last_stopped = ros::Time::now();
            car_stop_status.data = false;
        }
        else if((current_lane_uuid == old_lane_uuid) && ((now-time_when_last_stopped).toSec() > 10)){
            old_lane_uuid = current_lane_uuid;
            stopline_status.publish(car_stop_status);
            time_when_last_stopped = ros::Time::now();
            car_stop_status.data = false;
        }
    }
}

//Listens to higher level commands and plans accordingly
void plan_publisher::CommandNewPlanner(bool goal_reach_flg, bool cmd_changed_flag) {
    if(goal_reach_flg || cmd_changed_flag){
        geometry_msgs::PoseStamped pose_msg;
        raiscar_msgs::ControllerPath rais_control_msg;

        auto lane_plan = freicar::planning::lane_follower::GetPlan(start_point, command, 8, 40);

        // To publish on RVIZ
        PublishPlan(lane_plan, 0.0, 1.0, 0.0, 300, "plan_1", tf);

        // To publish on ROS
        rais_control_msg.path_segment.header.stamp = ros::Time::now();
        rais_control_msg.path_segment.header.frame_id = "map";
        rais_control_msg.path_segment.header.seq = 0;

        // To publish on ROS
            for(size_t i = 0; i < lane_plan.steps.size(); ++i) {
                pose_msg.pose.position.x = lane_plan.steps[i].position.x();
                pose_msg.pose.position.y = lane_plan.steps[i].position.y();
                pose_msg.pose.position.z = lane_plan.steps[i].position.z();
                /* Orientation used as a proxy for sending path description */
                pose_msg.pose.orientation.w = findPathDescription(lane_plan.steps[i].path_description);
                rais_control_msg.path_segment.poses.push_back(pose_msg);
            }
        path_to_follow.publish(rais_control_msg);

    }
}

int plan_publisher::findPathDescription(freicar::mapobjects::Lane::Connection description) {
    /*
     * 0 = JUNCTION_STRAIGHT: The next lane in a junction that goes straight
     * 1 = JUNCTION_LEFT: The next lane in a junction that turns left
     * 2 = JUNCTION_RIGHT: The next lane in a junction that turns right
     * 3 = STRAIGHT: The next lane that's not part of a junction
     * 4 = OPPOSITE: The opposite lane
     * 5 = ADJACENT_LEFT: The adjacent lane to the left
     * 6 = ADJACENT_RIGHT: The adjacent lane to the left
     * 7 = BACK: The previous lane
     * */
    int converted;
    switch(description){
        case freicar::mapobjects::Lane::JUNCTION_STRAIGHT:
            converted = 0;
            break;
        case freicar::mapobjects::Lane::JUNCTION_LEFT:
            converted = 1;
            break;
        case freicar::mapobjects::Lane::JUNCTION_RIGHT:
            converted = 2;
            break;
        case freicar::mapobjects::Lane::STRAIGHT:
            converted = 3;
            break;
        case freicar::mapobjects::Lane::OPPOSITE:
            converted = 4;
            break;
        case freicar::mapobjects::Lane::ADJACENT_LEFT:
            converted = 5;
            break;
        case freicar::mapobjects::Lane::ADJACENT_RIGHT:
            converted = 6;
            break;
        case freicar::mapobjects::Lane::BACK:
            converted = 7;
            break;
        default:
            converted = 3;
    }
    return converted;
}

//Plan to perform Overtake action.
std::vector<freicar::mapobjects::Point3D> plan_publisher::OvertakePlan(){
    //    const freicar::mapobjects::Lane* current_lane;
    auto& getInstance = freicar::map::Map::GetInstance();
    const freicar::mapobjects::Lane* opposite_lane;

    freicar::mapobjects::Point3D opp_3dp, next_3dp,next_prev_3dp;
    std::vector<freicar::mapobjects::Point3D> opp_3dps, next_3dps, final_3dps;

    std::vector<std::pair<freicar::mapobjects::LanePoint3D, float>> current_lp =
            getInstance.FindClosestLanePointsWithHeading(start_point.x(), start_point.y(), start_point.z(), 3, start_angle);

    std::vector<std::pair<freicar::mapobjects::LanePoint3D, float>> plan_current_lp =
            getInstance.FindClosestLanePointsWithHeading(start_point.x(), start_point.y(), start_point.z(), 100, start_angle);



    auto start_lp = current_lp[0].first;
    auto current_lane = getInstance.FindLaneByUuid(start_lp.GetLaneUuid());
    auto* lane_group = getInstance.FindLaneGroupByUuid(current_lane->GetParentUuid());

    if(!current_lane->IsJunctionLane()) {
        if(current_lane->GetLaneDirection() == 1) {        //Current Lane Direction is RIGHT
            for (auto& left_lane_uuid : lane_group->GetLeftLanes()) {
                opposite_lane = getInstance.FindLaneByUuid(left_lane_uuid);
            }
        }
        else if(current_lane->GetLaneDirection() == 2) {   //Current Lane Direction is LEFT
            for (auto& right_lane_uuid : lane_group->GetRightLanes()) {
                opposite_lane = getInstance.FindLaneByUuid(right_lane_uuid);
            }
        }

        auto prev_lane = getInstance.FindLaneByUuid(opposite_lane->GetOutConnections()[1]);
        auto prev_lane_size =  prev_lane->GetPoints().size();
        for(int i=prev_lane_size-1; i >= 0 ; i-- ){
            opp_3dp = prev_lane->GetPoints().at(i);
            opp_3dps.push_back(opp_3dp);
        }

        int lane_threshold = (int)(opposite_lane->GetPoints().size()*0.75);

        for(int i=opposite_lane->GetPoints().size()-1; i > 0 ; i-- ){
            opp_3dp = opposite_lane->GetPoints().at(i);
            opp_3dps.push_back(opp_3dp);
        }

        auto next_lane = getInstance.FindLaneByUuid(current_lane->GetOutConnections()[1]);

        int cutoff = 0;
        if( next_lane->GetPoints().size() > 9){
            cutoff = 5;
        }
        else if(next_lane->GetPoints().size() > 7){
            cutoff = 3;
        }
        for(int i=cutoff; i < next_lane->GetPoints().size(); ++i ){
            next_3dp = next_lane->GetPoints().at(i);
            next_3dps.push_back(next_3dp);
        }

/* Commented code
        int current_dir = current_lane->GetLaneDirection();
        std::vector<int> current_lp_points;
        for(int i=0; i < plan_current_lp.size(); i++ ){
            int point_dir = getInstance.FindLaneByUuid(plan_current_lp[i].first.GetLaneUuid())->GetLaneDirection();
            if(current_dir != point_dir){
                final_3dps.push_back(plan_current_lp[i].first);
            }
            if((current_dir == point_dir) && plan_current_lp[i].second > 3.0){
                current_lp_points.push_back(i);
            }
        }
        for(int i=0; i < current_lp_points.size(); i++ ){
            final_3dps.push_back(plan_current_lp[current_lp_points.at(i)].first);
        }
*/

        for(int i=0; i < opp_3dps.size(); i++ ){
            final_3dps.push_back(opp_3dps.at(i));
        }
        for(int i=0; i < next_3dps.size(); i++ ){
            final_3dps.push_back(next_3dps.at(i));
        }

        return final_3dps;
    }

}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "freicar_drive");

    freicar::map::ThriftMapProxy map_proxy("127.0.0.1", 9091, 9090);
    std::string map_path;

    if (!ros::param::get("/map_path", map_path)) {
        ROS_ERROR("could not find parameter: map_path! map initialization failed");
    }

// if the map can't be loaded
    if (!map_proxy.LoadMapFromFile(map_path)) {
        ROS_INFO("could not find thriftmap file: %s, starting map server...", map_path.c_str());
        map_proxy.StartMapServer();
        // stalling main thread until map is received
        while (freicar::map::Map::GetInstance().status() == freicar::map::MapStatus::UNINITIALIZED) {
//            ROS_INFO("waiting for map...", );
            ros::Duration(1.0).sleep();
        }
        ROS_INFO("map received!");
        // Thrift creats a corrupt file on failed reads, removing it
        remove(map_path.c_str());
        map_proxy.WriteMapToFile(map_path);
        ROS_INFO("saved new map");
    }

    freicar::map::Map::GetInstance().PostProcess(0.22);



    std::shared_ptr<ros::NodeHandle> n_ = std::make_shared<ros::NodeHandle>();
    plan_publisher pbb(n_);

    ros::spin();

    return 0;
}
