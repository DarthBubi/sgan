cmake_minimum_required(VERSION 3.0.2)
project(sgan_ros)

find_package(catkin REQUIRED COMPONENTS
  catkin_virtualenv
  geometry_msgs
  spencer_tracking_msgs
  visualization_msgs
)

catkin_generate_virtualenv(PYTHON_INTERPRETER python3.6)
catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES sgan_ros
#  CATKIN_DEPENDS Pillow geometry_msgs numpy spencer_tracking_msgs torch torchvision visualization_msgs
#  DEPENDS system_lib
)

#############
## Install ##
#############

catkin_install_python(PROGRAMS
  scripts/predict_tracks.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
