#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit_msgs/msg/constraints.hpp>
#include <moveit_msgs/msg/orientation_constraint.hpp>

#include "panda_moveit_planner/srv/plan_to_pose.hpp"

using PlanToPose = panda_moveit_planner::srv::PlanToPose;
using Trigger = std_srvs::srv::Trigger;

class PlannerServer : public rclcpp::Node
{
public:
    PlannerServer(const rclcpp::NodeOptions &options)
        : Node("panda_moveit_planner", options)
    {
        callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        service_ = create_service<PlanToPose>(
            "/plan_to_pose",
            std::bind(&PlannerServer::plan_callback, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default,
            callback_group_);

        execute_next_service_ = create_service<Trigger>(
            "/execute_next_plan",
            std::bind(&PlannerServer::execute_next_callback, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default,
            callback_group_);

        clear_queue_service_ = create_service<Trigger>(
            "/clear_plan_queue",
            std::bind(&PlannerServer::clear_queue_callback, this,
                      std::placeholders::_1, std::placeholders::_2),
            rmw_qos_profile_services_default,
            callback_group_);
    }

    void init()
    {
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), "panda_arm");

        move_group_->setEndEffectorLink("panda_link8");
        move_group_->setPlanningTime(5.0);
        move_group_->setNumPlanningAttempts(10);
        move_group_->setGoalPositionTolerance(0.0005);
        move_group_->setGoalOrientationTolerance(0.005);
        move_group_->startStateMonitor(2.0);

        robot_model_ = move_group_->getRobotModel();
        joint_group_ = robot_model_->getJointModelGroup("panda_arm");
        if (!joint_group_)
            throw std::runtime_error("Cannot find panda_arm JointModelGroup");

        RCLCPP_INFO(get_logger(), "============================================");
        RCLCPP_INFO(get_logger(), "MoveIt multi-IK queue planner ready");
        RCLCPP_INFO(get_logger(), "Planning group: panda_arm");
        RCLCPP_INFO(get_logger(), "End effector: panda_link8");
        RCLCPP_INFO(get_logger(), "Plan/cache service: /plan_to_pose");
        RCLCPP_INFO(get_logger(), "Execute next service: /execute_next_plan");
        RCLCPP_INFO(get_logger(), "Clear queue service: /clear_plan_queue");
        RCLCPP_INFO(get_logger(), "============================================");
    }

private:
    struct IKCandidate
    {
        std::vector<double> joints;
        double distance = 0.0;
    };

    double joint_distance(const std::vector<double> &a,
                          const std::vector<double> &b) const
    {
        double s = 0.0;
        const size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i)
        {
            const double d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }

    std::vector<IKCandidate> generate_ik_candidates(
        const moveit::core::RobotState &start_state,
        const geometry_msgs::msg::Pose &target_pose,
        int max_candidates = 20,
        int max_trials = 250)
    {
        std::vector<IKCandidate> candidates;
        std::vector<double> start_q;
        start_state.copyJointGroupPositions(joint_group_, start_q);

        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<double> local_noise(0.0, 0.45);

        for (int trial = 0;
             trial < max_trials && static_cast<int>(candidates.size()) < max_candidates;
             ++trial)
        {
            moveit::core::RobotState state(start_state);

            if (trial == 0)
            {
                state.setJointGroupPositions(joint_group_, start_q);
            }
            else if (trial < max_trials / 2)
            {
                // Search near the current/virtual posture first.
                std::vector<double> seed = start_q;
                for (double &v : seed)
                    v += local_noise(rng);
                state.setJointGroupPositions(joint_group_, seed);
                state.enforceBounds(joint_group_);
            }
            else
            {
                // Then broaden the seed search to discover other redundancy branches.
                state.setToRandomPositions(joint_group_);
            }

            state.update();

            const bool found = state.setFromIK(
                joint_group_, target_pose, "panda_link8", 0.08);

            if (!found)
                continue;

            state.enforceBounds(joint_group_);
            state.update();

            std::vector<double> q;
            state.copyJointGroupPositions(joint_group_, q);

            bool duplicate = false;
            for (const auto &old : candidates)
            {
                if (joint_distance(old.joints, q) < 0.04)
                {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate)
                continue;

            candidates.push_back({q, joint_distance(start_q, q)});
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const IKCandidate &a, const IKCandidate &b)
                  {
                      return a.distance < b.distance;
                  });

        return candidates;
    }

    bool plan_with_multiple_ik(
        const moveit::core::RobotState &start_state,
        const geometry_msgs::msg::PoseStamped &target,
        double velocity_scale,
        double acceleration_scale,
        moveit::planning_interface::MoveGroupInterface::Plan &selected_plan,
        size_t &selected_index,
        size_t &candidate_count)
    {
        auto candidates = generate_ik_candidates(start_state, target.pose, 20, 250);
        candidate_count = candidates.size();

        if (candidates.empty())
        {
            RCLCPP_ERROR(get_logger(), "No IK solution found for target pose");
            return false;
        }

        RCLCPP_INFO(get_logger(), "Generated %zu unique IK solutions", candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            const auto &c = candidates[i];
            RCLCPP_INFO(
                get_logger(),
                "IK[%zu] distance=%.4f q=[%.3f %.3f %.3f %.3f %.3f %.3f %.3f]",
                i, c.distance,
                c.joints.size() > 0 ? c.joints[0] : 0.0,
                c.joints.size() > 1 ? c.joints[1] : 0.0,
                c.joints.size() > 2 ? c.joints[2] : 0.0,
                c.joints.size() > 3 ? c.joints[3] : 0.0,
                c.joints.size() > 4 ? c.joints[4] : 0.0,
                c.joints.size() > 5 ? c.joints[5] : 0.0,
                c.joints.size() > 6 ? c.joints[6] : 0.0);
        }

        move_group_->setPoseReferenceFrame(target.header.frame_id);
        move_group_->setPlanningPipelineId("ompl");
        move_group_->setPlannerId("RRTConnectkConfigDefault");
        move_group_->setMaxVelocityScalingFactor(velocity_scale);
        move_group_->setMaxAccelerationScalingFactor(acceleration_scale);

        for (size_t i = 0; i < candidates.size(); ++i)
        {
            RCLCPP_INFO(get_logger(),
                        "Trying IK[%zu/%zu], distance=%.4f",
                        i + 1, candidates.size(), candidates[i].distance);

            move_group_->clearPoseTargets();
            move_group_->setStartState(start_state);

            if (!move_group_->setJointValueTarget(candidates[i].joints))
            {
                RCLCPP_WARN(get_logger(), "IK[%zu] rejected as joint target", i);
                continue;
            }

            moveit::planning_interface::MoveGroupInterface::Plan candidate_plan;
            const auto result = move_group_->plan(candidate_plan);

            if (result == moveit::core::MoveItErrorCode::SUCCESS)
            {
                selected_plan = candidate_plan;
                selected_index = i;
                RCLCPP_INFO(get_logger(), "Planning SUCCESS with IK[%zu]", i);
                return true;
            }

            RCLCPP_WARN(get_logger(), "Planning FAILED with IK[%zu]; trying next", i);
        }

        return false;
    }

    geometry_msgs::msg::Quaternion quaternion_from_start_state(
        const moveit::core::RobotState &start_state) const
    {
        const Eigen::Isometry3d &tf =
            start_state.getGlobalLinkTransform("panda_link8");

        Eigen::Quaterniond q(tf.rotation());
        q.normalize();

        geometry_msgs::msg::Quaternion out;
        out.x = q.x();
        out.y = q.y();
        out.z = q.z();
        out.w = q.w();
        return out;
    }

    bool plan_cartesian_locked_orientation(
        const moveit::core::RobotState &start_state,
        const geometry_msgs::msg::PoseStamped &requested_target,
        double velocity_scale,
        double acceleration_scale,
        moveit::planning_interface::MoveGroupInterface::Plan &plan,
        double &fraction)
    {
        // For a "linear" segment, preserve exactly the orientation at the
        // beginning of this segment. Only XYZ is taken from the request.
        geometry_msgs::msg::Pose waypoint = requested_target.pose;
        waypoint.orientation = quaternion_from_start_state(start_state);

        move_group_->clearPoseTargets();
        move_group_->clearPathConstraints();
        move_group_->setPoseReferenceFrame(requested_target.header.frame_id);
        move_group_->setStartState(start_state);
        move_group_->setMaxVelocityScalingFactor(velocity_scale);
        move_group_->setMaxAccelerationScalingFactor(acceleration_scale);

        std::vector<geometry_msgs::msg::Pose> waypoints{waypoint};
        moveit_msgs::msg::RobotTrajectory trajectory_msg;

        // Small Cartesian interpolation step to keep the tool path straight.
        constexpr double eef_step = 0.005;       // 5 mm
        constexpr double jump_threshold = 0.0;  // disable jump heuristic

        fraction = move_group_->computeCartesianPath(
            waypoints,
            eef_step,
            jump_threshold,
            trajectory_msg,
            false);  // collisions are currently ignored in this simulation

        if (fraction < 0.995 ||
            trajectory_msg.joint_trajectory.points.empty())
        {
            RCLCPP_WARN(
                get_logger(),
                "Cartesian path incomplete: fraction=%.3f",
                fraction);
            return false;
        }

        // computeCartesianPath gives a geometric path. Add timestamps so the
        // cached trajectory can be executed by the controller.
        robot_trajectory::RobotTrajectory rt(robot_model_, "panda_arm");
        rt.setRobotTrajectoryMsg(start_state, trajectory_msg);

        trajectory_processing::IterativeParabolicTimeParameterization iptp;
        const bool timed = iptp.computeTimeStamps(
            rt,
            velocity_scale,
            acceleration_scale);

        if (!timed)
        {
            RCLCPP_WARN(get_logger(), "Cartesian time parameterization failed");
            return false;
        }

        moveit_msgs::msg::RobotTrajectory timed_msg;
        rt.getRobotTrajectoryMsg(timed_msg);

        plan.trajectory_ = timed_msg;
        moveit::core::robotStateToRobotStateMsg(
            start_state,
            plan.start_state_);
        RCLCPP_INFO(
            get_logger(),
            "Cartesian locked-orientation planning SUCCESS, fraction=%.3f, points=%zu",
            fraction,
            plan.trajectory_.joint_trajectory.points.size());

        return true;
    }

    bool plan_orientation_constrained_ompl(
        const moveit::core::RobotState &start_state,
        geometry_msgs::msg::PoseStamped target,
        double velocity_scale,
        double acceleration_scale,
        moveit::planning_interface::MoveGroupInterface::Plan &selected_plan,
        size_t &selected_index,
        size_t &candidate_count)
    {
        // Keep the same orientation as the beginning of the segment.
        target.pose.orientation = quaternion_from_start_state(start_state);

        moveit_msgs::msg::OrientationConstraint oc;
        oc.header = target.header;
        oc.link_name = "panda_link8";
        oc.orientation = target.pose.orientation;
        oc.absolute_x_axis_tolerance = 0.01;
        oc.absolute_y_axis_tolerance = 0.01;
        oc.absolute_z_axis_tolerance = 0.01;
        oc.weight = 1.0;

        moveit_msgs::msg::Constraints constraints;
        constraints.name = "lock_panda_link8_orientation";
        constraints.orientation_constraints.push_back(oc);

        move_group_->setPathConstraints(constraints);

        const bool ok = plan_with_multiple_ik(
            start_state,
            target,
            velocity_scale,
            acceleration_scale,
            selected_plan,
            selected_index,
            candidate_count);

        move_group_->clearPathConstraints();

        return ok;
    }

    std::shared_ptr<moveit::core::RobotState>
    make_end_state(
        const moveit::core::RobotState& start_state,
        const moveit::planning_interface::MoveGroupInterface::Plan& plan)
    {
        auto state =
            std::make_shared<moveit::core::RobotState>(
                start_state);

        const auto& jt =
            plan.trajectory_.joint_trajectory;

        if (jt.points.empty())
        {
            return state;
        }

        const auto& last =
            jt.points.back();

        const auto& variable_names =
            robot_model_->getVariableNames();

        const size_t n =
            std::min(
                jt.joint_names.size(),
                last.positions.size());

        for (size_t i = 0; i < n; ++i)
        {
            const auto it = std::find(
                variable_names.begin(),
                variable_names.end(),
                jt.joint_names[i]);

            if (it != variable_names.end())
            {
                state->setVariablePosition(
                    jt.joint_names[i],
                    last.positions[i]);
            }
        }

        state->update();

        return state;
    }

    void plan_callback(
        const std::shared_ptr<PlanToPose::Request> request,
        std::shared_ptr<PlanToPose::Response> response)
    {
        std::lock_guard<std::mutex> lock(plan_mutex_);

        const auto &target = request->target;
        const double velocity_scale = std::clamp(request->velocity_scale, 0.01, 1.0);
        const double acceleration_scale = std::clamp(request->acceleration_scale, 0.01, 1.0);

        RCLCPP_INFO(get_logger(),
                    "Target frame=%s position=[%.6f %.6f %.6f], execute=%s",
                    target.header.frame_id.c_str(),
                    target.pose.position.x,
                    target.pose.position.y,
                    target.pose.position.z,
                    request->execute ? "true" : "false");

        if (request->linear)
            RCLCPP_INFO(
                get_logger(),
                "linear=true: trying Cartesian straight-line planning with locked tool orientation");

        std::shared_ptr<moveit::core::RobotState> start_state;

        if (!request->execute)
        {
            // Queue planning: first segment starts from the real robot, later segments
            // start from the endpoint of the previously cached trajectory.
            if (virtual_start_state_)
            {
                start_state = std::make_shared<moveit::core::RobotState>(*virtual_start_state_);
            }
            else
            {
                auto current = move_group_->getCurrentState(2.0);
                if (!current)
                {
                    response->success = false;
                    response->message = "Cannot read current robot state";
                    return;
                }
                start_state = std::make_shared<moveit::core::RobotState>(*current);
            }
        }
        else
        {
            auto current = move_group_->getCurrentState(2.0);
            if (!current)
            {
                response->success = false;
                response->message = "Cannot read current robot state";
                return;
            }
            start_state = std::make_shared<moveit::core::RobotState>(*current);
        }

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        size_t selected_index = 0;
        size_t candidate_count = 0;
        double cartesian_fraction = 0.0;

        bool ok = false;
        bool used_cartesian = false;
        bool used_constrained_fallback = false;

        if (request->linear)
        {
            ok = plan_cartesian_locked_orientation(
                *start_state,
                target,
                velocity_scale,
                acceleration_scale,
                plan,
                cartesian_fraction);

            used_cartesian = ok;

            if (!ok)
            {
                RCLCPP_WARN(
                    get_logger(),
                    "Cartesian planning failed/incomplete; falling back to "
                    "OMPL with panda_link8 orientation constraint");

                ok = plan_orientation_constrained_ompl(
                    *start_state,
                    target,
                    velocity_scale,
                    acceleration_scale,
                    plan,
                    selected_index,
                    candidate_count);

                used_constrained_fallback = ok;
            }
        }
        else
        {
            move_group_->clearPathConstraints();

            ok = plan_with_multiple_ik(
                *start_state,
                target,
                velocity_scale,
                acceleration_scale,
                plan,
                selected_index,
                candidate_count);
        }

        if (!ok)
        {
            // If any preplanning segment fails, invalidate the whole queue so that
            // no partially planned task can be executed accidentally.
            if (!request->execute)
            {
                planned_queue_.clear();
                virtual_start_state_.reset();
                RCLCPP_ERROR(get_logger(), "Preplanning failed: cached queue cleared");
            }

            response->success = false;
            response->message = request->linear
                ? "Linear Cartesian planning and orientation-constrained OMPL fallback both failed"
                : "All IK candidates failed planning";
            return;
        }

        if (!request->execute)
        {
            planned_queue_.push_back(plan);
            virtual_start_state_ = make_end_state(*start_state, plan);

            response->success = true;

            std::string planning_mode;
            if (used_cartesian)
                planning_mode = "Cartesian locked-orientation";
            else if (used_constrained_fallback)
                planning_mode = "OMPL orientation-constrained fallback";
            else
                planning_mode =
                    "OMPL multi-IK selected IK " +
                    std::to_string(selected_index) +
                    " of " + std::to_string(candidate_count);

            response->message =
                "Plan cached; mode=" + planning_mode +
                "; queue size=" + std::to_string(planned_queue_.size());

            RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
            return;
        }

        RCLCPP_INFO(get_logger(), "Immediate execution requested; executing selected plan");
        const auto execution_result = move_group_->execute(plan);
        if (execution_result != moveit::core::MoveItErrorCode::SUCCESS)
        {
            response->success = false;
            response->message = "Planning succeeded but execution failed";
            return;
        }

        response->success = true;
        response->message = "Planning and execution successful";
    }

    void execute_next_callback(
        const std::shared_ptr<Trigger::Request>,
        std::shared_ptr<Trigger::Response> response)
    {
        std::lock_guard<std::mutex> lock(plan_mutex_);

        if (planned_queue_.empty())
        {
            response->success = false;
            response->message = "No cached plan available";
            return;
        }

        RCLCPP_INFO(get_logger(), "Executing cached plan; remaining before execution=%zu",
                    planned_queue_.size());

        auto &plan = planned_queue_.front();
        const auto result = move_group_->execute(plan);

        if (result != moveit::core::MoveItErrorCode::SUCCESS)
        {
            response->success = false;
            response->message = "Cached trajectory execution failed";
            RCLCPP_ERROR(get_logger(), "%s", response->message.c_str());
            return;
        }

        planned_queue_.pop_front();
        response->success = true;
        response->message = "Cached trajectory executed; remaining=" +
                            std::to_string(planned_queue_.size());

        if (planned_queue_.empty())
            virtual_start_state_.reset();

        RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
    }

    void clear_queue_callback(
        const std::shared_ptr<Trigger::Request>,
        std::shared_ptr<Trigger::Response> response)
    {
        std::lock_guard<std::mutex> lock(plan_mutex_);
        const size_t old_size = planned_queue_.size();
        planned_queue_.clear();
        virtual_start_state_.reset();

        response->success = true;
        response->message = "Plan queue cleared; removed=" + std::to_string(old_size);
        RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
    }

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    moveit::core::RobotModelConstPtr robot_model_;
    const moveit::core::JointModelGroup *joint_group_ = nullptr;

    rclcpp::Service<PlanToPose>::SharedPtr service_;
    rclcpp::Service<Trigger>::SharedPtr execute_next_service_;
    rclcpp::Service<Trigger>::SharedPtr clear_queue_service_;
    rclcpp::CallbackGroup::SharedPtr callback_group_;

    std::deque<moveit::planning_interface::MoveGroupInterface::Plan> planned_queue_;
    std::shared_ptr<moveit::core::RobotState> virtual_start_state_;
    std::mutex plan_mutex_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);

    auto node = std::make_shared<PlannerServer>(options);
    node->init();

    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
