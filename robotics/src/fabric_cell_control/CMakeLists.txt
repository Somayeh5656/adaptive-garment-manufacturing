cmake_minimum_required(VERSION 3.8)
project(fabric_cell_control)

find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME}
  PACKAGE_DIR "fabric_cell_control"
  SCRIPTS
    fabric_cell_control/perception_node.py
    fabric_cell_control/cutting_service.py
)

ament_package()