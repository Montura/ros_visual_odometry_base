cmake_minimum_required(VERSION 2.8)
project(hello_world)

find_package(OpenCV 3 REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(src ${EIGEN3_INCLUDE_DIR})

find_package(catkin REQUIRED COMPONENTS
            roscpp
            rosbag
            sensor_msgs
            message_filters
            )

include_directories(include ${catkin_INCLUDE_DIRS}
                    CATKIN_DEPENDS roscpp rosbag sensor_msgs message_filters)

#catkin_package(DEPENDS eigen opencv)

add_executable(visual_odometry src/visual_odometry_base.cpp)
target_link_libraries(visual_odometry ${catkin_LIBRARIES} ${OpenCV_LIBS})

