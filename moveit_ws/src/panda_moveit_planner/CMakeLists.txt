cmake_minimum_required(VERSION 3.8)
project(panda_moveit_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# MoveIt Humble dependency compatibility
find_package(Boost REQUIRED COMPONENTS
  chrono
  date_time
  filesystem
  iostreams
  program_options
  random
  regex
  serialization
  system
  thread
)

find_package(eigen_stl_containers REQUIRED)
find_package(tl_expected REQUIRED)
find_package(random_numbers REQUIRED)

# Current tl_expected package exports tl_expected::tl_expected,
# while some Humble MoveIt exports still reference tl::expected.
if(TARGET tl_expected::tl_expected AND NOT TARGET tl::expected)
  add_library(tl::expected ALIAS tl_expected::tl_expected)
endif()

# Load MoveIt only after the compatibility targets above exist
find_package(moveit_ros_planning_interface REQUIRED)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  "srv/PlanToPose.srv"
  DEPENDENCIES geometry_msgs
)

add_executable(
  planner_server
  src/planner_server.cpp
)

ament_target_dependencies(
  planner_server
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
)

rosidl_get_typesupport_target(
  cpp_typesupport_target
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp"
)

target_link_libraries(
  planner_server
  "${cpp_typesupport_target}"
)

install(
  TARGETS planner_server
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(
  rosidl_default_runtime
)

ament_package()
