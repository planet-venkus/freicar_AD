/*
 * Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
 * Project: FreiCAR
 * Do NOT distribute this code to anyone outside the FreiCAR project
 */

#include "ros_vis.h"

#include <Eigen/Eigenvalues>
using namespace Eigen;
/*
 * This class comprises of many visualization functions that can send points or lines as ros msgs to rviz
 */

bool weightComparator(Particle a, Particle b){
    return (a.weight > b.weight);
}

geometry_msgs::Pose normalizeParticles(geometry_msgs::Pose pose_msg, int n){
    pose_msg.position.x = pose_msg.position.x/(n);
    pose_msg.position.y = pose_msg.position.y/(n);
    pose_msg.position.z = pose_msg.position.z/(n);
    return pose_msg;
}

ros_vis::ros_vis(std::shared_ptr<ros::NodeHandle> n):n_(n)
{
    marker_pub_ = n_->advertise<visualization_msgs::Marker>("particlep_filter_info_marker", 10);
    poses_pub_ = n_->advertise<geometry_msgs::PoseArray>("particles", 10);
    best_particle_pub_ = n_->advertise<geometry_msgs::PoseArray>("best_particle", 10);
    best_300_poses_pub_ = n_->advertise<geometry_msgs::PoseArray>("best300particles", 1);
}

void ros_vis::SendPoints(PointCloud<float> pts, const std::string ns, const std::string frame, float r, float g, float b){
    static size_t cnt = 0;
    visualization_msgs::Marker points;
    points.header.frame_id = frame;
    points.header.stamp =  ros::Time::now();
    points.ns = ns;
    points.action = visualization_msgs::Marker::ADD;
    points.pose.orientation.w = 1.0;
    points.id = 0;
    points.type = visualization_msgs::Marker::POINTS;
    points.scale.x = 0.1;
    points.scale.y = 0.1;
    points.scale.z = 0.1;

    points.color.r = r;
    points.color.g = g;
    points.color.b = b;
    points.color.a = 1.0;

    for(size_t i = 0; i < pts.pts.size(); i++){
        geometry_msgs::Point p;
        p.x = static_cast<double>(pts.pts.at(i).x);
        p.y = static_cast<double>(pts.pts.at(i).y);
        p.z = 0;
        points.points.push_back(p);
    }

    marker_pub_.publish(points);
    marker_pub_.publish(points);
}

void ros_vis::SendSigns(std::vector<Sign> signs, const std::string ns, const std::string frame){
    static size_t cnt = 0;
    visualization_msgs::Marker points;
    points.header.frame_id = frame;
    points.header.stamp =  ros::Time::now();
    points.ns = ns;
    points.action = visualization_msgs::Marker::ADD;
    points.pose.orientation.w = 1.0;
    points.id = 0;
    points.type = visualization_msgs::Marker::POINTS;
    points.scale.x = 0.1;
    points.scale.y = 0.1;
    points.scale.z = 0.1;

    points.color.r = 1.0;
    points.color.g = 0.5;
    points.color.b = 0.3;
    points.color.a = 1.0;

    for(size_t i = 0; i < signs.size(); i++){
        geometry_msgs::Point p;
        const Sign& s = signs.at(i);
        p.x = static_cast<double>(s.position[0]);
        p.y = static_cast<double>(s.position[1]);
        p.z = 0;
        points.points.push_back(p);
    }

    marker_pub_.publish(points);
    marker_pub_.publish(points);
}

void ros_vis::SendPoses(std::vector<Particle > poses, const std::string ns, const std::string frame){
    geometry_msgs::PoseArray poses_msg;

    poses_msg.header.frame_id = frame;
    poses_msg.header.stamp =  ros::Time::now();
    poses_msg.header.seq = 0;

    //    for(size_t i = 0; i < poses.size(); i++){
    for(auto p_i = poses.begin(); p_i != poses.end(); p_i++){
        Eigen::Transform<float,3,Eigen::Affine> transform = p_i->transform;

        geometry_msgs::Pose pose_msg;
        pose_msg.position.x = transform.translation().x();
        pose_msg.position.y = transform.translation().y();
        pose_msg.position.z = transform.translation().z();

        Eigen::Quaternionf rot(transform.rotation());
        pose_msg.orientation.w = rot.w();
        pose_msg.orientation.x = rot.x();
        pose_msg.orientation.y = rot.y();
        pose_msg.orientation.z = rot.z();

        poses_msg.poses.push_back(pose_msg);

    }
    poses_pub_.publish(poses_msg);
}


Eigen::Vector4d wavg_quaternion_markley(Eigen::MatrixXd Q, Eigen::VectorXd weights){
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    int M = Q.rows();
    double weightSum = 0;

    for(int i=0; i<M; i++){
        Eigen::Vector4d q = Q.row(i);
        if (q[0]<0)
            q = -q;
        A = weights[i]*q*q.adjoint() + A;
        weightSum += weights[i];
    }

    A=(1.0/weightSum)*A;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(A);
//    cout<<"A"<<endl<<A<<endl;
//    cout<<"vecs"<<endl<<eig.eigenvectors()<<endl;
//    cout<<"vals"<<endl<<eig.eigenvalues()<<endl;
    Vector4d qavg=eig.eigenvectors().col(3);
    return qavg;
}

void ros_vis::Send300BestPoses(std::vector<Particle > poses, const std::string ns, const std::string frame){
    geometry_msgs::PoseArray poses_msg;
    geometry_msgs::Pose pose_msg;
    std::vector<Particle> Best300Particles;

    int n = 300;
    int row = 0;

    poses_msg.header.frame_id = frame;
    poses_msg.header.stamp =  ros::Time::now();
    poses_msg.header.seq = 0;

    // Initialize matrices and vectors
    MatrixXd Quatern(n,4);
    VectorXd weight_vector(n,1);

    // Sort the weights
    std::sort(poses.begin(), poses.end(), weightComparator);

//    float total_weight;
    for(auto p_i = poses.begin(); p_i != poses.begin() + n; p_i++){
        Eigen::Transform<float,3,Eigen::Affine> transform = p_i->transform;
        float weight = p_i->weight;
        pose_msg.position.x += transform.translation().x();
        pose_msg.position.y += transform.translation().y();
        pose_msg.position.z += transform.translation().z();
        Eigen::Quaternionf rot(transform.rotation());

        // Push entries to weight_vector and quaternion matrix
        weight_vector(row,1) = weight;

        Quatern(row,0) = rot.w();
        Quatern(row,1) = rot.x();
        Quatern(row,2) = rot.y();
        Quatern(row,3) = rot.z();
        row++;
    }

    pose_msg = normalizeParticles(pose_msg, n);
    // Average quaternion
    Eigen::Vector4d qavg = wavg_quaternion_markley(Quatern, weight_vector);
    pose_msg.orientation.w = qavg[0];
    pose_msg.orientation.x = qavg[1];
    pose_msg.orientation.y = qavg[2];
    pose_msg.orientation.z = qavg[3];
    poses_msg.poses.push_back(pose_msg);

    best_300_poses_pub_.publish(poses_msg);
}

void ros_vis::SendBestParticle(const Particle& pose, const std::string frame){
    geometry_msgs::PoseArray poses_msg;

    poses_msg.header.frame_id = frame;
    poses_msg.header.stamp =  ros::Time::now();
    poses_msg.header.seq = 0;

    Eigen::Transform<float,3,Eigen::Affine> transform = pose.transform;

    geometry_msgs::Pose pose_msg;
    pose_msg.position.x = transform.translation().x();
    pose_msg.position.y = transform.translation().y();
    pose_msg.position.z = transform.translation().z();

    Eigen::Quaternionf rot(transform.rotation());
    pose_msg.orientation.w = rot.w();
    pose_msg.orientation.x = rot.x();
    pose_msg.orientation.y = rot.y();
    pose_msg.orientation.z = rot.z();

    poses_msg.poses.push_back(pose_msg);

    best_particle_pub_.publish(poses_msg);

}

void ros_vis::VisualizeDataAssociations(std::vector<Eigen::Vector3f> src, std::vector<Eigen::Vector3f> target, const std::string ns, const std::string frame, float r, float g, float b){
    static size_t cnt = 0;
    visualization_msgs::Marker points;
    points.header.frame_id = frame;
    points.header.stamp =  ros::Time::now();
    points.ns = ns;
    points.action = visualization_msgs::Marker::ADD;
    points.pose.orientation.w = 1.0;
    points.id = 0;
    points.type = visualization_msgs::Marker::LINE_LIST;
    points.scale.x = 0.04;
    points.scale.y = 0.04;
    points.scale.z = 0.04;

    points.color.r = r;
    points.color.g = g;
    points.color.b = b;
    points.color.a = 1.0;

    for(size_t i = 0; i < src.size(); i++){
        geometry_msgs::Point p;
        p.x = static_cast<double>(src.at(i).x());
        p.y = static_cast<double>(src.at(i).y());
        p.z = 0;
        points.points.push_back(p);

        geometry_msgs::Point q;
        q.x = static_cast<double>(target.at(i).x());
        q.y = static_cast<double>(target.at(i).y());
        q.z = 0;
        points.points.push_back(q);
    }

    marker_pub_.publish(points);
}

void ros_vis::VisualizeDataAssociations(std::vector<Sign> src, std::vector<Eigen::Vector3f> target, const std::string ns, const std::string frame, float r, float g, float b){
    static size_t cnt = 0;
    visualization_msgs::Marker points;
    points.header.frame_id = frame;
    points.header.stamp =  ros::Time::now();
    points.ns = ns;
    points.action = visualization_msgs::Marker::ADD;
    points.pose.orientation.w = 1.0;
    points.id = 0;
    points.type = visualization_msgs::Marker::LINE_LIST;
    points.scale.x = 0.04;
    points.scale.y = 0.04;
    points.scale.z = 0.04;

    points.color.r = r;
    points.color.g = g;
    points.color.b = b;
    points.color.a = 1.0;

    for(size_t i = 0; i < src.size(); i++){
        geometry_msgs::Point p;
        p.x = static_cast<double>(src.at(i).position.x());
        p.y = static_cast<double>(src.at(i).position.y());
        p.z = 0;
        points.points.push_back(p);

        geometry_msgs::Point q;
        q.x = static_cast<double>(target.at(i).x());
        q.y = static_cast<double>(target.at(i).y());
        q.z = 0;
        points.points.push_back(q);
    }

    marker_pub_.publish(points);
}


