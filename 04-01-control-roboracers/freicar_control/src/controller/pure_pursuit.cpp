/*
 * Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
 * Project: FreiCAR
 * Do NOT distribute this code to anyone outside the FreiCAR project
 */

/* A ROS implementation of the Pure pursuit path tracking algorithm (Coulter 1992).

   Terminology (mostly :) follows:
   Coulter, Implementation of the pure pursuit algoritm, 1992 and
   Sorniotti et al. Path tracking for Automated Driving, 2017.
 */

#include <string>
#include <cmath>
#include <algorithm>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/transform_datatypes.h>
#include <tf2/transform_storage.h>
#include <tf2/buffer_core.h>
#include <tf2/convert.h>
#include <tf2/utils.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <kdl/frames.hpp>
#include <raiscar_msgs/ControlReport.h>
#include "raiscar_msgs/ControlCommand.h"
#include "std_msgs/Bool.h"
#include "controller.h"

using std::string;

class PurePursuit: public controller
{
public:

    //! Constructor
    PurePursuit();

    //! Run the controller.
    void run();

private:
    void controller_step(nav_msgs::Odometry odom);
    float* getLookaheadPoint(float car_x, float car_y);
//    float default_ld;
};

float distance(float x2, float y2, float x1, float y1)
{
    /* Get eucledian distance between two points */
    float x_square = pow(x2-x1, 2);
    float y_square = pow(y2-y1, 2);
    return sqrt(x_square + y_square);
}

float signum(float x){
    if (x < 0)
        return -1;
    else
        return 1;
}

float* PurePursuit::getLookaheadPoint(float car_x, float car_y){
    /* @Param car_x x coordinate of car
     *  @Param car_y y coordinate of car
     *  @return float array containing lookahead point x, lookahead point y, intersection success as 0 or 1
     *  @see <a href="http://mathworld.wolfram.com/Circle-LineIntersection.html">Circle-Line Intersection</a>
     */
    float LookAheadPoint_x, LookAheadPoint_y = 0.0;
    bool intersection_success = false;
    static float lookahead[4];
    int intersection_idx;
    for(int i = 1; i < path_.size() ; i++){
        // Get the start and end point of the path segment in car frame
        float StartPoint_x = path_[i-1].getOrigin().x() - car_x;
        float StartPoint_y = path_[i-1].getOrigin().y() - car_y;
        float EndPoint_x = path_[i].getOrigin().x() - car_x;
        float EndPoint_y = path_[i].getOrigin().y() - car_y;

        // calculate an intersection of a segment and a circle with radius ld_dist_ (lookahead) and origin (0, 0)
        float dx = EndPoint_x - StartPoint_x;
        float dy = EndPoint_y - StartPoint_y;
        float d = distance(EndPoint_x, EndPoint_y, StartPoint_x, StartPoint_y);
        float D = (StartPoint_x * EndPoint_y) - (EndPoint_x * StartPoint_y);

        // if the discriminant is zero or the points are equal, there is no intersection
        float discriminant = (ld_dist_ * ld_dist_ * d * d) - (D * D);

        if((discriminant < 0) || ((StartPoint_x == EndPoint_x) && (StartPoint_y == EndPoint_y))){
            // no intersection
        }
        else{
            // the x components of the intersecting points
            float x1 = (float) ((D * dy) + (signum(dy) * dx * std::sqrt(discriminant))) / (d * d);
            float x2 = (float) ((D * dy) - (signum(dy) * dx * std::sqrt(discriminant))) / (d * d);

            // the y components of the intersecting points
            float y1 = (float) ((-D * dx) + (std::abs(dy) * std::sqrt(discriminant))) / (d * d);
            float y2 = (float) ((-D * dx) - (std::abs(dy) * std::sqrt(discriminant))) / (d * d);

            // whether each of the intersections are within the segment (and not the entire line)
            bool validIntersection1 = ((std::min(StartPoint_x, EndPoint_x) < x1) && (x1 < std::max(StartPoint_x, EndPoint_x)))
                                          || ((std::min(StartPoint_y, EndPoint_y) < y1) && (y1 < std::max(StartPoint_y, EndPoint_y)));
            bool validIntersection2 = ((std::min(StartPoint_x, EndPoint_x) < x2) && (x2 < std::max(StartPoint_x, EndPoint_x)))
                                      || ((std::min(StartPoint_y, EndPoint_y) < y2) && (y2 < std::max(StartPoint_y, EndPoint_y)));

            // remove the old lookahead if either of the points will be selected as the lookahead
            if (validIntersection1 || validIntersection2){
                LookAheadPoint_x = LookAheadPoint_y = 0.0;
                intersection_idx = i;
            }

            // select the first one if it's valid
            if (validIntersection1) {
                LookAheadPoint_x = x1 + car_x;
                LookAheadPoint_y = y1 + car_y;
                intersection_success = true;
            }

            // select the second one if it's valid and either lookahead is none,
            // or it's closer to the end of the segment than the first intersection
            if (validIntersection2) {
                if(((LookAheadPoint_x == 0) && (LookAheadPoint_y == 0)) ||
                    (std::abs(x1 - EndPoint_x) > std::abs(x2 - EndPoint_x)) ||
                        (std::abs(y1 - EndPoint_y) > std::abs(y2 - EndPoint_y))){
                    LookAheadPoint_x = x2 + car_x;
                    LookAheadPoint_y = y2 + car_y;
                    intersection_success = true;
                }
            }
        }
    }
    if(intersection_success){
        lookahead[0] = LookAheadPoint_x;
        lookahead[1] = LookAheadPoint_y;
        lookahead[2] = 1.0;
        lookahead[3] = (float)intersection_idx;
    }
    else{
        lookahead[0] = 0.0;
        lookahead[1] = 0.0;
        lookahead[2] = 0.0;
        lookahead[3] = 0.0;
    }
    return lookahead;
}

PurePursuit::PurePursuit()
{
    // Get parameters from the parameter server
    std::cout << "Pure Pursuit controller started..." << std::endl;
    current_gt_index = 0; /* Initialize ground truth path variable index to zero */
    number_of_retries = 0; /* Initialize number of retries for no intersection to zero */
    standing_vehicle_counter = 0; /* For Task4. To ensure that the car ahead is not moving */
}

/*
 * Implement your controller here! The function gets called each time a new odometry is incoming.
 * The path to follow is saved in the variable "path_". Once you calculated the new control outputs you can send it with
 * the pub_acker_ publisher.
 */
void PurePursuit::controller_step(nav_msgs::Odometry odom)
{
    // Code blocks that could be useful:
    geometry_msgs::TransformStamped tf_msg;
    geometry_msgs::TransformStamped front_axis_tf_msg;
    double remain_dist;
    tf2::Stamped<tf2::Transform> map_t_fa;
    try
    {
        tf_msg = tf_buffer_.lookupTransform(map_frame_id_, rear_axis_frame_id_, ros::Time(0));
        front_axis_tf_msg = tf_buffer_.lookupTransform(map_frame_id_, front_axis_frame_id_, ros::Time(0));
    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN_STREAM(ex.what());
    }

    /* Initialize variables
     * map_t_fa represents car's position and orientation as it is in rviz. It is set by getMeanParticle function in particle_filter.cpp
     * which sends a transform over to rviz consisting of position and orientation of best particle.
     * */
    tf2::convert(tf_msg, map_t_fa);
    double x = map_t_fa.getRotation().x();
    double y = map_t_fa.getRotation().y();
    double z = map_t_fa.getRotation().z();
    double w = map_t_fa.getRotation().w();
    double car_angle = atan2(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z));
    float car_x = map_t_fa.getOrigin().x(); // x co-ordinate of car position
    float car_y = map_t_fa.getOrigin().y(); // y co-ordinate of car position

    /* Calculating the Lookahead points */
    float* lookahead = getLookaheadPoint(car_x, car_y); // Replace with AdaptivePurePursuit if needed in future
    float curvature;
    float LookAheadPoint_x, LookAheadPoint_y;
    int intersection_success;
    int intersection_idx;
    LookAheadPoint_x = lookahead[0];
    LookAheadPoint_y = lookahead[1];
    intersection_success = (int)lookahead[2];
    intersection_idx = (int)lookahead[3];


    /* Fetch an index one point ahead of the last path index -- from the ground truth plan and follow that OR
     * Fetch new plan if no lookahead point found */
    bool fetch_new_plan = false;
    if(!intersection_success){
        if((path_.size() > 0) && current_gt_index+1 < path_.size() && number_of_retries < 6){
            if(number_of_retries == 0)
                current_gt_index += 1;
            LookAheadPoint_x = path_[current_gt_index].getOrigin().x();
            LookAheadPoint_y = path_[current_gt_index].getOrigin().y();
            number_of_retries += 1;
        }
        else{
            number_of_retries = 0;
            fetch_new_plan = true;
        }
    }
    else{
        current_gt_index = intersection_idx;
    }

    /* Code block for adaptive control
     * Finds the curvature and sets the desired velocity based on the curvature(More the curvature, less the speed)
     * */
    float a = -tan(car_angle);
    float b = 1;
    float c = tan(car_angle)*car_x - car_y;
    float denom =  sqrt(pow(a,2) + pow(b,2));
    float numer = std::abs( a*LookAheadPoint_x + b*LookAheadPoint_y + c );
    float x_dist = numer/denom;
    curvature = (2*x_dist)/pow(ld_dist_,2);
    des_v_ = std::min(vmax_, (curvature_vel_limit_factor/curvature) );


    /* The following code block can be used to control a certain velocity using PID control */
    float pid_vel_out = 0.0;
    if (des_v_ >= 0) {
        pid_vel_out = vel_pid.step((des_v_ - odom.twist.twist.linear.x), ros::Time::now());
    } else {
        pid_vel_out = des_v_;
        vel_pid.resetIntegral();
    }

    // Check if the path to follow is received and the car is not yet finished following the complete path.
    if((path_.size() > 0) && (stp_resume_cmd != "stop") && (car_stop_status == false) && rightOfWay_status){

        /* Future Prediction
         * Override velocity if there is no concensus in the next x points
         * */
        int current_description = (int)_path_desc[intersection_idx-1];
        bool override_vel = false;
        for(int i = 0; i <= 4; i++){
            if((intersection_idx < _path_desc.size()-4) && (current_description != (int)_path_desc[intersection_idx+i])){
                override_vel = true;
                break;
            }
        }
        if(override_vel){
            des_v_ = 0.05;
        }

        /* Code for visualizing the lookahead point */
        if(intersection_success) {
            projection_t_.transform.translation.x = LookAheadPoint_x;
            projection_t_.transform.translation.y = LookAheadPoint_y;
            projection_t_.transform.translation.z = 0.0;

            projection_t_.transform.rotation.x = 0.0;
            projection_t_.transform.rotation.y = 0.0;
            projection_t_.transform.rotation.z = 0.0;
            projection_t_.transform.rotation.w = 1.0;
//            projection_t_.header.frame_id = rear_axis_frame_id_;
            projection_t_.header.frame_id = map_frame_id_;
            projection_t_.header.stamp = odom.header.stamp;
            tf_broadcaster_.sendTransform(projection_t_);
        }

        /* Calculating the steering angle for car */
        float alpha = atan2((LookAheadPoint_y - car_y), (LookAheadPoint_x - car_x)) - car_angle;
        current_steering_angle_ = std::min( atan2((2.0 * L_ * sin(alpha)) , ld_dist_/steering_vel_limit_factor), delta_max_ );

        /* Check distance to the waypoint where we need to update our plan
         * For smooth car following, start generating plan when we are midway through the plan instead of last point.
         * */
        int path_generating_wp = (int) (path_.size()-4); // Path generating waypoint
//		if(IsVehicleStanding)
//			path_generating_wp = (int) (path_.size()-2);
        remain_dist = distance(car_x, car_y,
                                      path_[path_generating_wp].getOrigin().x(), path_[path_generating_wp].getOrigin().y());

        /* The following code block sends out the boolean true to signal that the last waypoint is reached */
        if(remain_dist < pos_tol_){
            goal_reached_ = true;
            sendGoalMsg(true); // Send goal message to fetch a new plan
        }

    }
    /* Goal is initially reached when the car is at its initial spawn location and when there is still no path to follow
     * If the path is not received or the robocar has finished traversing through the given path or detected stop sign
     * Set angle and velocity to zero
     * */
    else if ((path_.size() == 0 || fetch_new_plan) && (!IsVehicleStanding)){
        // TODO Add a condition which checks whether the velocities and all should be zero OR a new plan should be fetched!
        // This will replace a timeout
        current_steering_angle_ = 0;
        pid_vel_out = 0;
        completion_advertised_ = false;
        sendGoalMsg(true);
    }
    else if(stp_resume_cmd == "stop") // || car_stopsign
    {
        pid_vel_out = 0;
    }
    else if (car_stop_status){
        cmd_control_.brake = 0.5;
    }
	else if(!rightOfWay_status){
		cmd_control_.brake = 0.7;
	}

    if((IsVehicleStanding)&&(!Request_Overtake_Plan)){

        current_steering_angle_ = 0;
        pid_vel_out = 0;
        cmd_control_.brake = 0.8;
        standing_vehicle_counter += 1;
        if(standing_vehicle_counter > 200){
            Request_Overtake_Plan = true;
            std_msgs::Bool msg;
            msg.data = Request_Overtake_Plan;
            pub_overtake_request_.publish(msg);
            standing_vehicle_counter = 0;
        }
    }
    else if((path_.size() > 0) && IsVehicleStanding){
        // Release brake and follow plan
        cmd_control_.brake = 0.0;
    }
    else if((!IsVehicleStanding) && !car_stop_status && rightOfWay_status ){
        /* reset variables */
        cmd_control_.brake = 0.0;
        standing_vehicle_counter = 0;
        Request_Overtake_Plan = false;
    }

    cmd_control_.steering = current_steering_angle_ / (70 * M_PI/180.0); //  DUMMY_STEERING_ANGLE should be a value in degree
    cmd_control_.throttle =  pid_vel_out;
    cmd_control_.throttle_mode =  0;
    cmd_control_.throttle = std::min(cmd_control_.throttle, throttle_limit_);
    cmd_control_.throttle = std::max(std::min((double)cmd_control_.throttle, 1.0), 0.0);
    /* logic for reversing in overtake situation */
//    if(standing_vehicle_counter > 160 && standing_vehicle_counter < 200){
//        cmd_control_.steering -= 0.5;
//        std::cout << "Negative throttle " << std::endl;
//        cmd_control_.brake = 0.0;
//        cmd_control_.throttle = -0.1;
//    }
    pub_acker_.publish(cmd_control_);

    /* Reset the car stop status and stop line pose */
    if(car_stop_status == true){
        ros::Duration(5).sleep();
        car_stop_status = false;
        cmd_control_.brake = 0;
        des_v_ = 0;
    }
    // Release break when we have right of way
    if(!rightOfWay_status){
        ros::Duration(10).sleep();
        rightOfWay_status = true;
        cmd_control_.brake = 0;
        des_v_ = 0;
    }
}

void PurePursuit::run()
{
    ros::spin();
}

int main(int argc, char**argv)
{
    ros::init(argc, argv, "pure_pursuit_controller");

    PurePursuit controller;
    controller.run();

    return 0;
}
