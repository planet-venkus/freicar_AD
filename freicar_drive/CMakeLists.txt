#cmake_minimum_required(VERSION 2.8.3)
#project(freicar_drive)
#
#
#set(CMAKE_BUILD_TYPE Release)
#
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
#set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -fPIC")
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#
#set(CMAKE_CXX_STANDARD 11)
#
### Compile as C++11, supported in ROS Kinetic and newer
## add_compile_options(-std=c++11)
#
### Find catkin macros and libraries
### if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
### is used, also find other catkin packages
#find_package(Threads REQUIRED)
#find_package(Threads REQUIRED)
#find_package(Boost)
#find_package(nanoflann)
#find_package (Eigen3 REQUIRED NO_MODULE)
#
#find_program(LSB_RELEASE_EXEC lsb_release)
#execute_process(COMMAND ${LSB_RELEASE_EXEC} -ds
#    OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
#    OUTPUT_STRIP_TRAILING_WHITESPACE
#)
#
#if (${LSB_RELEASE_ID_SHORT} STREQUAL "Ubuntu 20.04.1 LTS")
#        message("Ubuntu 20.04 detected...")
#        find_package(OpenCV REQUIRED)
#else()
#        message("Ubuntu < 20.04 detected....")
#        find_package(OpenCV 3 REQUIRED)
#endif()
#
#
#find_package(catkin REQUIRED COMPONENTS
#        roscpp
#        rospy
#        tf
#        freicar_sign_detect
#        sensor_msgs
#        std_msgs
#        nav_msgs
#        geometry_msgs
#        visualization_msgs
#        image_transport
#        cv_bridge
#        freicar_common
#        freicar_map
#        tf_conversions
#)
#
#
### System dependencies are found with CMake's conventions
## find_package(Boost REQUIRED COMPONENTS system)
#
#catkin_package(
##  INCLUDE_DIRS include
##  LIBRARIES freicar_sign_detect
#  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs freicar_sign_detect freicar_common freicar_map tf_conversions
##  DEPENDS system_lib
#)
#include_directories(
## include
#  ${catkin_INCLUDE_DIRS}
#  ${Boost_INCLUDE_DIR}
#  ${EIGEN_INC}
#  ${THRIFT_INCLUDE_DIR}
#)
#
##add_library(
#    #    ros_map
#    #    src/sensor_model.cpp
#    #    src/ros_vis.cpp
#    #    src/particle_filter.cpp
#    #    src/map_helper.cpp
#    #)
## Eigen3::Eigen removed from libs
## target_link_libraries(
#    # ros_map
#                     # ${OpenCV_LIBS}
#    # ${Boost_LIBRARIES}
#                      # ${THRIFT_LIBRARIES}
#                                       # ${CMAKE_THREAD_LIBS_INIT}
#                                       # Eigen3::Eigen
#                       # yaml-cpp
#                      # )
#
#
#add_executable(${PROJECT_NAME}_node src/freicar_odometry.cpp)
#
#add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})
#
### Specify libraries to link a library or executable target against
#target_link_libraries(${PROJECT_NAME}_node
#   nanoflann::nanoflann
#   #  ros_map
#   ${catkin_LIBRARIES}
#   ${CMAKE_THREAD_LIBS_INIT}
#   pthread
#   Eigen3::Eigen
#        yaml-cpp
#)

cmake_minimum_required(VERSION 2.8.3)
project(freicar_drive)

#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

# your C++ compilation flags
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# finding non-ROS libraries, e.g. Eigen
find_package (Eigen3 3 REQUIRED)

# finding ROS packages
find_package(catkin REQUIRED COMPONENTS
        # required for using ROS in C++
        image_boundingboxinfo_publisher
        roscpp
        # other ros packages, e.g.
        std_msgs
        tf2
        tf2_ros
        # the local ROS package you want
        freicar_map
        raiscar_msgs
        )
# dependecies of the current package
catkin_package(CATKIN_DEPENDS
        image_boundingboxinfo_publisher
        roscpp
        std_msgs
        tf2
        tf2_ros
        freicar_map
        raiscar_msgs
        )
# header files to consider
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

# your executable ROS node
add_executable(${PROJECT_NAME}_node src/plan_publisher.cpp src/plan_publisher.h)
add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES}
        Eigen3::Eigen
        )

