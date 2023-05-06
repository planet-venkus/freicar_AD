//
// Created by and code implemented by Venkat as part of Teamroboracers from 3/19/21.
//

#include "agent_camera.h"

agent_camera::agent_camera(std::shared_ptr<ros::NodeHandle> n): n_(n){
    rgb_sub = n_->subscribe("/freicar_1/sim/camera/rgb/front/image", 10, &agent_camera::RGBCallback, this);
    depth_sub = n_->subscribe("/freicar_1/sim/camera/depth/front/image_float", 10, &agent_camera::DepthCallback,this);
    bb_sub = n_->subscribe("bbsarray", 10, &agent_camera::BoundingBox, this);
    car_ahead_pub_ = n_->advertise<std_msgs::Bool>("car_ahead", 1);

    //Used in camera intrinsic matrix mulatiplication
    intrinsic_mat << fx , 0  , cx,
                      0 , fx , cy,
                      0 , 0  ,  1;

    intrinsic_mat_inv = intrinsic_mat.inverse();
}
void agent_camera::RGBCallback(const sensor_msgs::ImageConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat rgb8_img = cv_ptr->image;
    cv::Mat rgb_reize;
    cv::resize(rgb8_img,rgb_reize, cv::Size(640, 384));
}

void agent_camera::DepthCallback(const sensor_msgs::ImageConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
    }
	catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat float32_img = cv_ptr->image;
    cv::Mat f32_img_resize, uchar_img, uchar_img_scaled;

    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;

    std::vector<float> z_min_dist_vect;
    std_msgs::Bool car_ahead_msg;
    bool car_ahead_flag;


    Eigen::Vector3d camera_coord;
//    cv::imshow("depth listner image", float32_img);
    cv::resize(float32_img,f32_img_resize, cv::Size(640, 384));

    for(auto i=0; i < bounding_boxes_.bounding_boxes.size();i++){

        double z_min_dist = 1000.0;
        image_boundingboxinfo_publisher::box bounding_box = bounding_boxes_.bounding_boxes.at(i);

        float width_x = bounding_box.xmax - bounding_box.xmin;
        float height_y = bounding_box.ymax - bounding_box.ymin;
        float z_dist = f32_img_resize.at<float>(cv::Point(bounding_box.xcenter, bounding_box.ycenter));
        Eigen::Vector3d image_coord(bounding_box.xcenter, bounding_box.ycenter, 1);
        cv::Rect roi(bounding_box.xmin,bounding_box.ymin,width_x,height_y);
        cv::Mat img_bb_cropped = f32_img_resize(roi);

        cv::minMaxLoc(img_bb_cropped, &minVal, &maxVal, &minLoc, &maxLoc);
//        std::cout << "minVal: " << minVal << ", maxVal: " << maxVal << ", minLoc: " << minLoc << ", maxLoc: " << maxLoc << "\n";

        if(minVal <= car_ahead_threshold){
            car_ahead_flag = true;
            car_ahead_msg.data = car_ahead_flag;
            car_ahead_pub_.publish(car_ahead_msg);
        }
        else{
            car_ahead_flag = false;
            car_ahead_msg.data = car_ahead_flag;
            car_ahead_pub_.publish(car_ahead_msg);
        }
        camera_coord = z_dist * intrinsic_mat_inv * image_coord;

        z_min_dist_vect.push_back(z_min_dist);

        camera_to_map_proj_.transform.translation.x = camera_coord(0);
        camera_to_map_proj_.transform.translation.y = camera_coord(1);
        camera_to_map_proj_.transform.translation.z = camera_coord(2);

        camera_to_map_proj_.transform.rotation.x = 0.0;
        camera_to_map_proj_.transform.rotation.y = 0.0;
        camera_to_map_proj_.transform.rotation.z = 0.0;
        camera_to_map_proj_.transform.rotation.w = 1.0;
        camera_to_map_proj_.header.frame_id = "freicar_1/zed_camera";
        camera_to_map_proj_.child_frame_id = "freicar_1/depth_point_" + std::to_string(i);
        camera_to_map_proj_.header.stamp = ros::Time::now();
        tf_broadcaster_.sendTransform(camera_to_map_proj_);

    }
//    cv::waitKey(2);

}

void agent_camera::BoundingBox(const image_boundingboxinfo_publisher::boxesPtr &msg) {
    bounding_boxes_ = *msg;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "agent_camera");

    ROS_INFO("AGENT_CAMERA Node started");

    std::shared_ptr<ros::NodeHandle> n_ = std::make_shared<ros::NodeHandle>();

    agent_camera agt(n_); //object of agent camera


    ros::spin();

    return 0;

}