cmake_minimum_required(VERSION 3.10)
project(legged_wbc)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED
        COMPONENTS
        roscpp
        legged_interface
        qpoases_catkin
        )

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(PkgConfig REQUIRED)
pkg_check_modules(pinocchio REQUIRED pinocchio)

find_package(OsqpEigen REQUIRED)

# Please comment this line if you don't have eigen version problem. If you want to use it, you need to change the eigen include path
set(pinocchio_INCLUDE_DIRS "/home/bmtsz/NMPC_WBC_ws/devel/include;/opt/ros/noetic/include;/usr/include/eigen3")

catkin_package(
        INCLUDE_DIRS
        include
        LIBRARIES
        ${PROJECT_NAME}
        CATKIN_DEPENDS
        roscpp
        DEPENDS
        pinocchio
)

###########
## Build ##
###########

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${pinocchio_INCLUDE_DIRS}
)

link_directories(
        ${pinocchio_LIBRARY_DIRS}
)

add_library(${PROJECT_NAME}
        src/ho_qp.cpp
        src/wbc.cpp
        src/projected_wbc.cpp
        )

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        # OsqpEigen::OsqpEigen
        )

target_compile_options(${PROJECT_NAME} PUBLIC ${OCS2_CXX_FLAGS})

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

# Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h"
        )

#############
## Testing ##
#############
if (CATKIN_ENABLE_TESTING)
    catkin_add_gtest(${PROJECT_NAME}_test
            test/ho_qp_test.cpp
            )

    target_link_libraries(${PROJECT_NAME}_test
            ${PROJECT_NAME}
            ${pinocchio_LIBRARIES}
            gtest_main
            )
endif (CATKIN_ENABLE_TESTING)
