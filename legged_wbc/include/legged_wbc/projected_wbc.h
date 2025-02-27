//
// Created by Shengzhi Wang on 2022/10/22.
//

#pragma once

#include "legged_wbc/ho_qp.h"
#include "math/pseudoInverse.h"

#include <legged_interface/legged_interface.h>
#include <ocs2_legged_robot/gait/MotionPhaseDefinition.h>
#include <ocs2_centroidal_model/PinocchioCentroidalDynamics.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>

// OSQP Eigen
#include "OsqpEigen/OsqpEigen.h"
namespace legged
{
using namespace ocs2;
using namespace legged_robot;

// Decision Variables: x = [\tau]
class ProjectedWBC
{
public:
  ProjectedWBC(const std::string& task_file, LeggedInterface& legged_interface,
      const PinocchioEndEffectorKinematics& ee_kinematics, bool verbose);
  vector_t update(const vector_t& state_desired, const vector_t& input_desired, vector_t& measured_rbd_state,
                  size_t mode);

private:
  void loadTasksSetting(const std::string& task_file, bool verbose);
  void getFootContactJacobian();
  void getFootFrictionConeConstraint();
  void getManipulability(const matrix_t& J);
  void calculateRobotDynamics(const pinocchio::DataTpl<scalar_t, 0, pinocchio::JointCollectionDefaultTpl>& data);
  void calculateReferenceTorque(size_t mode);
  void getQPSoftAndHardConstraint_noArm();
  vector_t getHQPSolution();

  Task formulateFloatingBaseEomTask();
  Task formulateTorqueLimitsTask();
  Task formulateNoContactMotionTask();
  Task formulateFrictionConeTask();
  Task formulateBaseAccelTask();
  Task formulateSwingLegTask();
  Task formulateContactForceTask();

  size_t num_decision_vars_;
  PinocchioInterface& pino_interface_;
  const CentroidalModelInfo& info_;
  PinocchioCentroidalDynamics centroidal_dynamics_;
  CentroidalModelPinocchioMapping mapping_;
  std::unique_ptr<PinocchioEndEffectorKinematics> ee_kinematics_;

  vector_t state_desired_, input_desired_, measured_q_, measured_v_;
  matrix_t j_, dj_;   // stack foot jacobian and the derivativeof the stack jacobian
  contact_flag_t contact_flag_;
  size_t num_contacts_;
  size_t contact_num_for_support_, contact_num_for_force_control_;

  // Definitions from Shengzhi's previous work
  matrix_t Jacobian_contact_, d_Jacobian_contact_;
  matrix_t Jacobian_contact_support_;  // Jcs
  matrix_t Jacobian_contact_force_control_;  // Jcf
  matrix_t Jacobian_base_, d_Jacobian_base_;  //Jb, d_Jb
  matrix_t pinv_Jacobian_contact_;
  matrix_t FrictionConstraint_matrix_; // C matrix
  matrix_t FrictionConstraint_for_support_matrix_; // C_s matrix
  matrix_t FrictionConstraint_for_force_control_matrix_; // C_f matrix
  matrix_t Projection_matrix_;  // P
  matrix_t d_projection_matrix_;  // dP
  matrix_t Inertial_matrix_, L_, M_bar_, Lambda_, inv_Lambda_;
  matrix_t Hessian_matrix_;
  matrix_t Constraint_matrix_;
  matrix_t Selection_matrix_; // B (selection matrix)
  matrix_t Selection_matrix_support_; // B_s (selection matrix for the actuators used to support the torso)
  matrix_t Selection_matrix_force_; // B_f (selection matrix for the actuators used for the force control)
  vector_t low_d_vector_, high_d_vector_;
  vector_t low_D_m_vector_, high_D_m_vector_;
  vector_t low_D_f_vector_, high_D_f_vector_;
  vector_t h_vector_, h_bar_;
  vector_t lastDesiredVelocity_pinocchio_actuatedJoints_;
  vector_t torque_command_;
  vector_t v_mrt_prev_;
  vector_t f_vector_;
  vector_t F_base_ext_est_;
  vector_t lowerBound_, upperBound_;
  vector_t TorqueLowerBound_, TorqueUpperBound_;

  scalar_t manipulability_;

  // Task Parameters:
  vector_t torque_limits_;
  scalar_t friction_coeff_, swing_kp_, swing_kd_;
  scalar_t friction_coef_; // Note: It is specificaly used for Shengzhi's WBC controller design and this is different from "friction_coeff_" above.

  // Impedance Control parameters
  scalar_t task_P_gain_foot = 7000;
  scalar_t task_D_gain_foot = 35;
  scalar_t task_P_gain_base_ = 2000;
  scalar_t task_D_gain_base_ = 100;

  // OSQP solver
  OsqpEigen::Solver solver;

  // time to activate the contact force control
  double count_;
  double waitTime_;
};

}  // namespace legged
