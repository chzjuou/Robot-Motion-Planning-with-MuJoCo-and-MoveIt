#include <algorithm>
#include <memory>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#include "panda_moveit_planner/srv/plan_to_pose.hpp"

using PlanToPose = panda_moveit_planner::srv::PlanToPose;

class PlannerServer : public rclcpp::Node
{
public:
    PlannerServer(const rclcpp::NodeOptions &options)
        : Node("panda_moveit_planner", options)
    {
        callback_group_ = create_callback_group(
            rclcpp::CallbackGroupType::Reentrant);

        service_ = create_service<PlanToPose>(
            "/plan_to_pose",
            std::bind(
                &PlannerServer::plan_callback,
                this,
                std::placeholders::_1,
                std::placeholders::_2),
            rmw_qos_profile_services_default,
            callback_group_);
    }

    void init()
    {
        move_group_ = std::make_shared<
            moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(),
                "panda_arm");

        move_group_->setEndEffectorLink("panda_link8");
        move_group_->setPlanningTime(5.0);
        move_group_->setNumPlanningAttempts(10);
        move_group_->setGoalPositionTolerance(0.0005);
        move_group_->setGoalOrientationTolerance(0.005);
        move_group_->startStateMonitor(2.0);

        RCLCPP_INFO(get_logger(), "============================================");
        RCLCPP_INFO(get_logger(), "MoveIt planner server ready");
        RCLCPP_INFO(get_logger(), "Planning group: panda_arm");
        RCLCPP_INFO(get_logger(), "End effector: panda_link8");
        RCLCPP_INFO(get_logger(), "Service: /plan_to_pose");
        RCLCPP_INFO(get_logger(), "============================================");
    }

private:
    void plan_callback(
        const std::shared_ptr<PlanToPose::Request> request,
        std::shared_ptr<PlanToPose::Response> response)
    {
        std::lock_guard<std::mutex> lock(plan_mutex_);

        const auto &target = request->target;

        double velocity_scale = std::clamp(
            request->velocity_scale,
            0.01,
            1.0);

        double acceleration_scale = std::clamp(
            request->acceleration_scale,
            0.01,
            1.0);

        RCLCPP_INFO(
            get_logger(),
            "Target frame=%s position=[%.6f %.6f %.6f]",
            target.header.frame_id.c_str(),
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z);

        RCLCPP_INFO(
            get_logger(),
            "Velocity scale=%.3f, acceleration scale=%.3f",
            velocity_scale,
            acceleration_scale);

        // ============================================================
        // Start state = current MuJoCo /joint_states
        // ============================================================
        move_group_->setStartStateToCurrentState();
        move_group_->setPoseReferenceFrame(
            target.header.frame_id);

        // ============================================================
        // Planner
        // ============================================================
        if (request->linear)
        {
            RCLCPP_INFO(
                get_logger(),
                "Planner = Pilz LIN");

            move_group_->setPlanningPipelineId(
                "pilz_industrial_motion_planner");

            move_group_->setPlannerId("LIN");
        }
        else
        {
            RCLCPP_INFO(
                get_logger(),
                "Planner = OMPL RRTConnect");

            move_group_->setPlanningPipelineId(
                "ompl");

            move_group_->setPlannerId(
                "RRTConnectkConfigDefault");
        }

        move_group_->setMaxVelocityScalingFactor(
            velocity_scale);

        move_group_->setMaxAccelerationScalingFactor(
            acceleration_scale);

        // ============================================================
        // Target
        // ============================================================
        const bool pose_ok =
            move_group_->setPoseTarget(
                target.pose,
                "panda_link8");

        if (!pose_ok)
        {
            response->success = false;
            response->message =
                "MoveIt rejected pose target";

            move_group_->clearPoseTargets();
            return;
        }

        // ============================================================
        // Planning
        // ============================================================
        moveit::planning_interface::
            MoveGroupInterface::Plan plan;

        const auto planning_result =
            move_group_->plan(plan);

        move_group_->clearPoseTargets();

        if (planning_result !=
            moveit::core::MoveItErrorCode::SUCCESS)
        {
            RCLCPP_ERROR(
                get_logger(),
                "Planning failed");

            response->success = false;
            response->message =
                "MoveIt planning failed";

            return;
        }

        const auto &trajectory =
            plan.trajectory_.joint_trajectory;

        RCLCPP_INFO(
            get_logger(),
            "Planning successful, points=%zu",
            trajectory.points.size());

        if (!trajectory.points.empty())
        {
            const auto &last =
                trajectory.points.back();

            double duration =
                static_cast<double>(
                    last.time_from_start.sec) +
                static_cast<double>(
                    last.time_from_start.nanosec) * 1e-9;

            RCLCPP_INFO(
                get_logger(),
                "Trajectory duration=%.3f s",
                duration);
        }

        // ============================================================
        // Planning only
        // ============================================================
        if (!request->execute)
        {
            response->success = true;
            response->message =
                "Planning successful; execution disabled";
            return;
        }

        // ============================================================
        // Execute in MuJoCo
        // ============================================================
        RCLCPP_INFO(
            get_logger(),
            "Sending trajectory to MuJoCo...");

        const auto execution_result =
            move_group_->execute(plan);

        if (execution_result !=
            moveit::core::MoveItErrorCode::SUCCESS)
        {
            response->success = false;
            response->message =
                "Planning succeeded but execution failed";

            RCLCPP_ERROR(
                get_logger(),
                "Trajectory execution failed");

            return;
        }

        response->success = true;
        response->message =
            "Planning and execution successful";

        RCLCPP_INFO(
            get_logger(),
            "MuJoCo execution completed");
    }

    std::shared_ptr<
        moveit::planning_interface::MoveGroupInterface>
        move_group_;

    rclcpp::Service<
        PlanToPose>::SharedPtr
        service_;

    rclcpp::CallbackGroup::SharedPtr
        callback_group_;

    std::mutex plan_mutex_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;

    options
        .automatically_declare_parameters_from_overrides(
            true);

    auto node =
        std::make_shared<PlannerServer>(
            options);

    node->init();

    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(),
        4);

    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();

    return 0;
}
