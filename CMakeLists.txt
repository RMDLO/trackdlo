cmake_minimum_required(VERSION 3.0.2)
project(trackdlo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3)   # -std=c++11

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  pcl_conversions
	pcl_ros
)

add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PCL 1.8 REQUIRED COMPONENTS common io filters visualization features kdtree)
include_directories(include SYSTEM PUBLIC
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES tracking_ros
#  CATKIN_DEPENDS abb_egm_hardware_interface abb_egm_state_controller abb_rws_service_provider abb_rws_state_publisher controller_manager joint_state_controller velocity_controllers
#  DEPENDS system_lib
)

add_executable(
  trackdlo trackdlo/src/trackdlo_node.cpp trackdlo/src/trackdlo.cpp trackdlo/src/utils.cpp
)
target_link_libraries(trackdlo
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
)
# target_compile_options(trackdlo PRIVATE -O3 -Wall -Wextra -Wconversion -Wshadow -g)

add_executable(
  evaluation trackdlo/src/run_evaluation.cpp trackdlo/src/trackdlo.cpp trackdlo/src/utils.cpp trackdlo/src/evaluator.cpp
)
target_link_libraries(evaluation
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
)
# target_compile_options(evaluation PRIVATE -O3 -Wall -Wextra -Wconversion -Wshadow -g)

# add_executable(
#   cv_test trackdlo/src/test.cpp
# )
# target_link_libraries(cv_test
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
#   ${OpenCV_LIBS}
#   Eigen3::Eigen
# )
# target_compile_options(cv_test PRIVATE -O3 -Wall -Wextra -Wconversion -Wshadow -g)