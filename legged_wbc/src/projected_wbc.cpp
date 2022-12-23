//
// Created by Shengzhi Wang on 2022/10/22.
//
#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "legged_wbc/projected_wbc.h"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>

namespace legged
{
ProjectedWBC::ProjectedWBC(const std::string& task_file, LeggedInterface& legged_interface,
         const PinocchioEndEffectorKinematics& ee_kinematics, bool verbose)
  : pino_interface_(legged_interface.getPinocchioInterface())
  , info_(legged_interface.getCentroidalModelInfo())
  , centroidal_dynamics_(info_)
  , mapping_(info_)
  , ee_kinematics_(ee_kinematics.clone())
{
  num_decision_vars_ = info_.generalizedCoordinatesNum + 3 * info_.numThreeDofContacts + info_.actuatedDofNum;  // Original code
  // num_decision_vars_ = info_.actuatedDofNum;  // TODO: Here I will use the selection matrix B as nonsquare.
  centroidal_dynamics_.setPinocchioInterface(pino_interface_);
  mapping_.setPinocchioInterface(pino_interface_);
  measured_q_ = vector_t(info_.generalizedCoordinatesNum);
  measured_v_ = vector_t(info_.generalizedCoordinatesNum);
  ee_kinematics_->setPinocchioInterface(pino_interface_);

  loadTasksSetting(task_file, verbose);

  // Initialization of matrices
  Projection_matrix_.setZero(info_.generalizedCoordinatesNum, info_.generalizedCoordinatesNum);
  d_projection_matrix_.setZero(info_.generalizedCoordinatesNum, info_.generalizedCoordinatesNum);
  Inertial_matrix_.setZero(info_.generalizedCoordinatesNum, info_.generalizedCoordinatesNum);
  h_vector_.setZero(info_.generalizedCoordinatesNum);
  L_.setZero(info_.generalizedCoordinatesNum, info_.generalizedCoordinatesNum);
  M_bar_.setZero(info_.generalizedCoordinatesNum, info_.generalizedCoordinatesNum);
  h_bar_.setZero(info_.generalizedCoordinatesNum);
  Jacobian_base_.setZero(6, info_.generalizedCoordinatesNum);
  d_Jacobian_base_.setZero(6, info_.generalizedCoordinatesNum);
  Lambda_.setZero(6, 6);
  inv_Lambda_.setZero(6, 6);
  lastDesiredVelocity_pinocchio_actuatedJoints_.setZero(info_.actuatedDofNum);
}

vector_t ProjectedWBC::update(const vector_t& state_desired, const vector_t& input_desired, vector_t& measured_rbd_state,
                     size_t mode)
{
  state_desired_ = state_desired;
  input_desired_ = input_desired;

  contact_flag_ = modeNumber2StanceLeg(mode);
  num_contacts_ = 0;
  for (bool flag : contact_flag_)
    if (flag)
      num_contacts_++;

  measured_q_.head<3>() = measured_rbd_state.segment<3>(3);
  measured_q_.segment<3>(3) = measured_rbd_state.head<3>();
  measured_q_.tail(info_.actuatedDofNum) = measured_rbd_state.segment(6, info_.actuatedDofNum);

  measured_v_.head<3>() = measured_rbd_state.segment<3>(info_.generalizedCoordinatesNum + 3);
  measured_v_.segment<3>(3) = getEulerAnglesZyxDerivativesFromGlobalAngularVelocity<scalar_t>(
      measured_q_.segment<3>(3), measured_rbd_state.segment<3>(info_.generalizedCoordinatesNum));
  measured_v_.tail(info_.actuatedDofNum) =
      measured_rbd_state.segment(info_.generalizedCoordinatesNum + 6, info_.actuatedDofNum);

  const auto& model = pino_interface_.getModel();
  auto& data = pino_interface_.getData();

  // For floating base Cartesian Impedance Control
  pinocchio::forwardKinematics(model, data, measured_q_, measured_v_);
  pinocchio::computeJointJacobians(model, data);
  pinocchio::updateFramePlacements(model, data);
  pinocchio::crba(model, data, measured_q_);
  data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();
  pinocchio::nonLinearEffects(model, data, measured_q_, measured_v_);
  j_ = matrix_t(3 * info_.numThreeDofContacts, info_.generalizedCoordinatesNum);
  for (size_t i = 0; i < info_.numThreeDofContacts; ++i)
  {
    Eigen::Matrix<scalar_t, 6, Eigen::Dynamic> jac;
    jac.setZero(6, info_.generalizedCoordinatesNum);
    pinocchio::getFrameJacobian(model, data, info_.endEffectorFrameIndices[i], pinocchio::LOCAL_WORLD_ALIGNED, jac);
    j_.block(3 * i, 0, 3, info_.generalizedCoordinatesNum) = jac.template topRows<3>();
  }
  Jacobian_base_.setZero();
  pinocchio::getFrameJacobian(model, data, model.getFrameId("base"), 
                              pinocchio::LOCAL_WORLD_ALIGNED, Jacobian_base_);

  // For not contact motion task
  pinocchio::computeJointJacobiansTimeVariation(model, data, measured_q_, measured_v_);
  dj_ = matrix_t(3 * info_.numThreeDofContacts, info_.generalizedCoordinatesNum);
  for (size_t i = 0; i < info_.numThreeDofContacts; ++i)
  {
    Eigen::Matrix<scalar_t, 6, Eigen::Dynamic> jac;
    jac.setZero(6, info_.generalizedCoordinatesNum);
    pinocchio::getFrameJacobianTimeVariation(model, data, info_.endEffectorFrameIndices[i],
                                             pinocchio::LOCAL_WORLD_ALIGNED, jac);
    dj_.block(3 * i, 0, 3, info_.generalizedCoordinatesNum) = jac.template topRows<3>();
  }
  d_Jacobian_base_.setZero();
  pinocchio::getFrameJacobianTimeVariation(model, data, model.getFrameId("base"), 
                                           pinocchio::LOCAL_WORLD_ALIGNED, d_Jacobian_base_);

  // For base acceleration task
  updateCentroidalDynamics(pino_interface_, info_, measured_q_);

  // Get contact feet's Jacobian and Jacobian derivatives
  getFootContactJacobian();

  // Get the manipulability of the Jacobian_contact_
  getManipulability(Jacobian_contact_);

  // Calculate Robot Dynamics
  calculateRobotDynamics(data);

  // Calculate Reference Torque
  calculateReferenceTorque(mode);

  // Optimization problem
  Task task_0 = formulateFloatingBaseEomTask() + formulateTorqueLimitsTask() + formulateFrictionConeTask() +
                formulateNoContactMotionTask();
  Task task_1 = formulateBaseAccelTask() + formulateSwingLegTask();
  Task task_2 = formulateContactForceTask();
  HoQp ho_qp(task_2, std::make_shared<HoQp>(task_1, std::make_shared<HoQp>(task_0)));

  return ho_qp.getSolutions();
}

Task ProjectedWBC::formulateFloatingBaseEomTask()
{
  auto& data = pino_interface_.getData();

  matrix_t s(info_.actuatedDofNum, info_.generalizedCoordinatesNum);
  s.block(0, 0, info_.actuatedDofNum, 6).setZero();
  s.block(0, 6, info_.actuatedDofNum, info_.actuatedDofNum).setIdentity();

  matrix_t a(info_.generalizedCoordinatesNum, num_decision_vars_);
  vector_t b(info_.generalizedCoordinatesNum);
  a << data.M, -j_.transpose(), -s.transpose();
  b = -data.nle;
  return Task(a, b, matrix_t(), vector_t());
}

Task ProjectedWBC::formulateTorqueLimitsTask()
{
  matrix_t d(2 * info_.actuatedDofNum, num_decision_vars_);
  d.setZero();
  matrix_t i = matrix_t::Identity(info_.actuatedDofNum, info_.actuatedDofNum);
  d.block(0, info_.generalizedCoordinatesNum + 3 * info_.numThreeDofContacts, info_.actuatedDofNum,
          info_.actuatedDofNum) = i;
  d.block(info_.actuatedDofNum, info_.generalizedCoordinatesNum + 3 * info_.numThreeDofContacts, info_.actuatedDofNum,
          info_.actuatedDofNum) = -i;
  vector_t f(2 * info_.actuatedDofNum);
  for (size_t l = 0; l < 2 * info_.actuatedDofNum / 3; ++l)
    f.segment<3>(3 * l) = torque_limits_;

  return Task(matrix_t(), vector_t(), d, f);
}

Task ProjectedWBC::formulateNoContactMotionTask()
{
  matrix_t a(3 * num_contacts_, num_decision_vars_);
  vector_t b(a.rows());
  a.setZero();
  b.setZero();
  size_t j = 0;
  for (size_t i = 0; i < info_.numThreeDofContacts; i++)
    if (contact_flag_[i])
    {
      a.block(3 * j, 0, 3, info_.generalizedCoordinatesNum) = j_.block(3 * i, 0, 3, info_.generalizedCoordinatesNum);
      b.segment(3 * j, 3) = -dj_.block(3 * i, 0, 3, info_.generalizedCoordinatesNum) * measured_v_;
      j++;
    }

  return Task(a, b, matrix_t(), vector_t());
}

Task ProjectedWBC::formulateFrictionConeTask()
{
  matrix_t a(3 * (info_.numThreeDofContacts - num_contacts_), num_decision_vars_);
  a.setZero();
  size_t j = 0;
  for (size_t i = 0; i < info_.numThreeDofContacts; ++i)
    if (!contact_flag_[i])
      a.block(3 * j++, info_.generalizedCoordinatesNum + 3 * i, 3, 3) = matrix_t::Identity(3, 3);
  vector_t b(a.rows());
  b.setZero();

  matrix_t friction_pyramic(5, 3);
  friction_pyramic << 0, 0, -1, 1, 0, -friction_coeff_, -1, 0, -friction_coeff_, 0, 1, -friction_coeff_, 0, -1,
      -friction_coeff_;

  matrix_t d(5 * num_contacts_ + 3 * (info_.numThreeDofContacts - num_contacts_), num_decision_vars_);
  d.setZero();
  j = 0;
  for (size_t i = 0; i < info_.numThreeDofContacts; ++i)
    if (contact_flag_[i])
      d.block(5 * j++, info_.generalizedCoordinatesNum + 3 * i, 5, 3) = friction_pyramic;
  vector_t f = Eigen::VectorXd::Zero(d.rows());

  return Task(a, b, d, f);
}

Task ProjectedWBC::formulateBaseAccelTask()
{
  matrix_t a(6, num_decision_vars_);
  a.setZero();
  a.block(0, 0, 6, 6) = matrix_t::Identity(6, 6);

  const vector_t momentum_rate =
      info_.robotMass * centroidal_dynamics_.getValue(0, state_desired_, input_desired_).head(6);
  const Eigen::Matrix<scalar_t, 6, 6> a_base = getCentroidalMomentumMatrix(pino_interface_).template leftCols<6>();
  const auto a_base_inv = computeFloatingBaseCentroidalMomentumMatrixInverse(a_base);
  vector_t b = a_base_inv * momentum_rate;

  const vector3_t angular_velocity = getGlobalAngularVelocityFromEulerAnglesZyxDerivatives<scalar_t>(
      measured_q_.segment<3>(3), measured_v_.segment<3>(3));
  b.segment<3>(3) -= a_base_inv.block<3, 3>(3, 3) * angular_velocity.cross(a_base.block<3, 3>(3, 3) * angular_velocity);

  return Task(a, b, matrix_t(), vector_t());
}

Task ProjectedWBC::formulateSwingLegTask()
{
  std::vector<vector3_t> pos_measured = ee_kinematics_->getPosition(vector_t());
  std::vector<vector3_t> vel_measured = ee_kinematics_->getVelocity(vector_t(), vector_t());
  vector_t q_desired = mapping_.getPinocchioJointPosition(state_desired_);
  vector_t v_desired = mapping_.getPinocchioJointVelocity(state_desired_, input_desired_);
  const auto& model = pino_interface_.getModel();
  auto& data = pino_interface_.getData();
  pinocchio::forwardKinematics(model, data, q_desired, v_desired);
  pinocchio::updateFramePlacements(model, data);
  std::vector<vector3_t> pos_desired = ee_kinematics_->getPosition(vector_t());
  std::vector<vector3_t> vel_desired = ee_kinematics_->getVelocity(vector_t(), vector_t());

  matrix_t a(3 * (info_.numThreeDofContacts - num_contacts_), num_decision_vars_);
  vector_t b(a.rows());
  a.setZero();
  b.setZero();
  size_t j = 0;
  for (size_t i = 0; i < info_.numThreeDofContacts; ++i)
    if (!contact_flag_[i])
    {
      vector3_t accel = swing_kp_ * (pos_desired[i] - pos_measured[i]) + swing_kd_ * (vel_desired[i] - vel_measured[i]);
      a.block(3 * j, 0, 3, info_.generalizedCoordinatesNum) = j_.block(3 * i, 0, 3, info_.generalizedCoordinatesNum);
      b.segment(3 * j, 3) = accel - dj_.block(3 * i, 0, 3, info_.generalizedCoordinatesNum) * measured_v_;
      j++;
    }

  return Task(a, b, matrix_t(), vector_t());
}

Task ProjectedWBC::formulateContactForceTask()
{
  matrix_t a(3 * info_.numThreeDofContacts, num_decision_vars_);
  vector_t b(a.rows());
  a.setZero();

  for (size_t i = 0; i < info_.numThreeDofContacts; ++i)
    a.block(3 * i, info_.generalizedCoordinatesNum + 3 * i, 3, 3) = matrix_t::Identity(3, 3);
  b = input_desired_.head(a.rows());

  return Task(a, b, matrix_t(), vector_t());
}

void ProjectedWBC::loadTasksSetting(const std::string& task_file, bool verbose)
{
  // Load task file
  torque_limits_ = vector_t(info_.actuatedDofNum / 4);
  loadData::loadEigenMatrix(task_file, "torqueLimitsTask", torque_limits_);
  if (verbose)
  {
    std::cerr << "\n #### Torque Limits Task: \n";
    std::cerr << "\n #### =============================================================================\n";
    std::cerr << "\n #### HAA HFE KFE: " << torque_limits_.transpose() << "\n";
    std::cerr << " #### =============================================================================\n";
  }
  boost::property_tree::ptree pt;
  boost::property_tree::read_info(task_file, pt);
  std::string prefix = "frictionConeTask.";
  if (verbose)
  {
    std::cerr << "\n #### Friction Cone Task: ";
    std::cerr << "\n #### =============================================================================\n";
  }
  loadData::loadPtreeValue(pt, friction_coeff_, prefix + "frictionCoefficient", verbose);
  if (verbose)
  {
    std::cerr << " #### =============================================================================\n";
  }
  prefix = "swingLegTask.";
  if (verbose)
  {
    std::cerr << "\n #### Swing Leg Task: ";
    std::cerr << "\n #### =============================================================================\n";
  }
  loadData::loadPtreeValue(pt, swing_kp_, prefix + "kp", verbose);
  loadData::loadPtreeValue(pt, swing_kd_, prefix + "kd", verbose);
}

void ProjectedWBC::getFootContactJacobian()
{
  int Jc_count_ = 0;
  if (num_contacts_ != 0){
    Jacobian_contact_.setZero(3 * num_contacts_, info_.generalizedCoordinatesNum); d_Jacobian_contact_.setZero(3 * num_contacts_, info_.generalizedCoordinatesNum);
    for (size_t i = 0; i < info_.numThreeDofContacts; ++i)
      if (contact_flag_[i]){
        Jacobian_contact_.block(3 * Jc_count_, 0, 3, info_.generalizedCoordinatesNum) = j_.block(3 * i, 0, 3, info_.generalizedCoordinatesNum);
        d_Jacobian_contact_.block(3 * Jc_count_, 0, 3, info_.generalizedCoordinatesNum) = dj_.block(3 * i, 0, 3, info_.generalizedCoordinatesNum);
        Jc_count_++;
      }
    
    FrictionConstraint_matrix_.setZero(5 * num_contacts_, 3 * num_contacts_);
    low_d_vector_.setZero(5 * num_contacts_);
    high_d_vector_.setZero(5 * num_contacts_);
  }  
  else{
    Jacobian_contact_.setZero(3 * info_.numThreeDofContacts, info_.generalizedCoordinatesNum); 
    d_Jacobian_contact_.setZero(3 * info_.numThreeDofContacts, info_.generalizedCoordinatesNum);          
  }

  // Jcs: Jacobian of contact foot for supporting the torso exclusively (TODO: hardcoded)
  int Jcs_count_ = 0;
  int contact_num_for_support_ = contact_flag_[1] + contact_flag_[2] + contact_flag_[3];
  if (contact_num_for_support_ != 0){
    Jacobian_contact_support_.setZero(3 * contact_num_for_support_, info_.generalizedCoordinatesNum); 
    for (size_t i = 1; i < info_.numThreeDofContacts; ++i)
      if (contact_flag_[i]){
        Jacobian_contact_support_.block(3 * Jcs_count_, 0, 3, info_.generalizedCoordinatesNum) = j_.block(3 * i, 0, 3, info_.generalizedCoordinatesNum);
        Jcs_count_++;
      }
    
    FrictionConstraint_for_support_matrix_.setZero(5 * contact_num_for_support_, 3 * contact_num_for_support_);
    low_D_m_vector_.setZero(5 * contact_num_for_support_);
    high_D_m_vector_.setZero(5 * contact_num_for_support_);
  }  
  else{
    Jacobian_contact_support_.setZero(3 * (info_.numThreeDofContacts - 1), info_.generalizedCoordinatesNum); // d_Jacobian_contact_support_.setZero();           
  }    

  // Jcf: Jacobian of contact foot for the force control exclusively (TODO: hardcoded)
  int Jcf_count_ = 0;
  int contact_num_for_force_control_ = contact_flag_[0];
  if (contact_num_for_force_control_ != 0){
    Jacobian_contact_force_control_.setZero(3 * contact_num_for_force_control_, info_.generalizedCoordinatesNum); // d_Jacobian_contact_force_control_.setZero();  
    for (size_t i = 0; i < 1; ++i)
      if (contact_flag_[i]){
        Jacobian_contact_force_control_.block(3 * Jcf_count_, 0, 3, info_.generalizedCoordinatesNum) = j_.block(3 * i, 0, 3, info_.generalizedCoordinatesNum);
        Jcf_count_++;
      }

    FrictionConstraint_for_force_control_matrix_.setZero(5 * contact_num_for_force_control_, 3 * contact_num_for_force_control_);
    low_D_f_vector_.setZero(5 * contact_num_for_force_control_);
    high_D_f_vector_.setZero(5 * contact_num_for_force_control_);    
  }  
  else{
    Jacobian_contact_force_control_.setZero(3, info_.generalizedCoordinatesNum); // d_Jacobian_contact_force_control_.setZero();           
  }   
}

void ProjectedWBC::getManipulability(const matrix_t& J)
{
  manipulability_ = std::sqrt((J * J.transpose()).determinant());   // Manipulability
}

void ProjectedWBC::calculateRobotDynamics(const pinocchio::DataTpl<scalar_t, 0, pinocchio::JointCollectionDefaultTpl>& data)
{
  pinv_Jacobian_contact_.setZero(Jacobian_contact_.cols(), Jacobian_contact_.rows());
  pinv_Jacobian_contact_ = pseudoInverse(Jacobian_contact_);         //J_c_pinv 
  if (num_contacts_ != 0 && manipulability_ < 0.001){
    pinv_Jacobian_contact_.setZero();
    pinv_Jacobian_contact_ = Jacobian_contact_.transpose() * (Jacobian_contact_ * Jacobian_contact_.transpose() 
                            + std::pow(0.001, 2) * Eigen::MatrixXd::Identity(Jacobian_contact_.rows(), Jacobian_contact_.rows())).inverse(); //J_c_pinv (damped solution)
  }
  Projection_matrix_.setZero();
  Projection_matrix_ = Eigen::MatrixXd::Identity(info_.generalizedCoordinatesNum, info_.generalizedCoordinatesNum) - pinv_Jacobian_contact_ * Jacobian_contact_;   // P

  // Robot dynamics
  Inertial_matrix_.setZero();
  Inertial_matrix_ = data.M;
  h_vector_.setZero();
  h_vector_ = data.nle;

  // Projection matrix's time derivative: dp = L + L', L = -J_c_pinv * d_J_c * P
  L_.setZero();
  L_ = -pinv_Jacobian_contact_ * d_Jacobian_contact_ * Projection_matrix_;
  d_projection_matrix_.setZero();
  d_projection_matrix_ = L_ + L_.transpose();

  // Constrained joint space dynamics
  M_bar_.setZero();
  M_bar_ = Projection_matrix_ * Inertial_matrix_ + Eigen::MatrixXd::Identity(info_.generalizedCoordinatesNum, info_.generalizedCoordinatesNum) - Projection_matrix_;
  h_bar_.setZero();
  h_bar_ = Projection_matrix_ * h_vector_ - d_projection_matrix_ * measured_v_;

  // Constrained task space dynamics
  Lambda_.setZero();
  Lambda_ = (Jacobian_base_ * M_bar_.inverse() * Projection_matrix_ * Jacobian_base_.transpose()).inverse(); 
  // Lambda_ = getSVDBasedInverse(Jacobian_base_ * M_bar_.inverse() * Projection_matrix_ * Jacobian_base_.transpose()); 
  inv_Lambda_.setZero();
  inv_Lambda_ = Jacobian_base_ * M_bar_.inverse() * Projection_matrix_ * Jacobian_base_.transpose();  
}

void ProjectedWBC::calculateReferenceTorque(size_t mode)
{
  const auto& model = pino_interface_.getModel();
  auto& data = pino_interface_.getData();
  std::vector<vector3_t> pos_measured = ee_kinematics_->getPosition(vector_t());
  std::vector<vector3_t> vel_measured = ee_kinematics_->getVelocity(vector_t(), vector_t());
  const auto q_desired = mapping_.getPinocchioJointPosition(state_desired_);
  const auto v_desired = mapping_.getPinocchioJointVelocity(state_desired_, input_desired_);

  updateCentroidalDynamics(pino_interface_, info_, q_desired);

  // Base Pose in world frame
  const auto basePose_desired = q_desired.head<6>();
  const auto basePosition_desired = basePose_desired.head<3>();
  const auto baseOrientation_desired = basePose_desired.tail<3>();

  // Base Velocity in world frame
  const auto A = getCentroidalMomentumMatrix(pino_interface_);
  const Eigen::Matrix<scalar_t, 6, 6> Ab = A.template leftCols<6>();
  const auto Ab_inv = computeFloatingBaseCentroidalMomentumMatrixInverse(Ab);
  const auto Aj = A.rightCols(info_.actuatedDofNum);

  Eigen::Matrix<scalar_t, 6, 1> baseVelocity_desired;
  baseVelocity_desired.head<3>() = v_desired.head<3>();
  const auto derivativeEulerAnglesZyx_desired = v_desired.segment<3>(3);
  const auto baseAngularVelocityInWorld_ = getGlobalAngularVelocityFromEulerAnglesZyxDerivatives<scalar_t>(baseOrientation_desired, derivativeEulerAnglesZyx_desired);
  baseVelocity_desired.tail<3>() = baseAngularVelocityInWorld_;

  const auto Adot_ = pinocchio::dccrba(model, data, q_desired, v_desired);
  const double dt = 0.001;  // TODO: This might be changed later
  const vector_t jointAccelerations = (v_desired.tail(info_.actuatedDofNum) - lastDesiredVelocity_pinocchio_actuatedJoints_) / dt;
  vector_t centroidalMomentumRate = info_.robotMass * centroidal_dynamics_.getValue(0, state_desired_, input_desired_).head(6);
  centroidalMomentumRate.noalias() -= Adot_ * v_desired;
  centroidalMomentumRate.noalias() -= Aj * jointAccelerations.head(info_.actuatedDofNum);
  Eigen::Matrix<scalar_t, 6, 1> qbaseDdot_;
  qbaseDdot_.noalias() = Ab_inv * centroidalMomentumRate;

  // Base Acceleration in world frame (classical acceleration)
  Eigen::Matrix<scalar_t, 6, 1> baseDesAcceleration_classical;
  baseDesAcceleration_classical.head<3>() = qbaseDdot_.head<3>();
  const auto baseAngularAccelerationInWorld_ =
      getGlobalAngularAccelerationFromEulerAnglesZyxDerivatives<scalar_t>(baseOrientation_desired, derivativeEulerAnglesZyx_desired, qbaseDdot_.tail<3>());
  baseDesAcceleration_classical.tail<3>() = baseAngularAccelerationInWorld_;

  // renew last desired joint velocity
  lastDesiredVelocity_pinocchio_actuatedJoints_ = v_desired.tail(info_.actuatedDofNum);

  Eigen::VectorXd a_desired(info_.generalizedCoordinatesNum);
  a_desired << qbaseDdot_, jointAccelerations;
  pinocchio::forwardKinematics(model, data, q_desired, v_desired, a_desired);
  pinocchio::updateFramePlacements(model, data);
  std::vector<vector3_t> pos_desired = ee_kinematics_->getPosition(vector_t());
  std::vector<vector3_t> vel_desired = ee_kinematics_->getVelocity(vector_t(), vector_t());
  std::vector<vector3_t> acc_desired;
  for (const auto& frameId : info_.endEffectorFrameIndices) {
    acc_desired.emplace_back(pinocchio::getFrameClassicalAcceleration(model, data, frameId, pinocchio::LOCAL_WORLD_ALIGNED).linear());
  }

  if (mode == ocs2::legged_robot::ModeNumber::STANCE){
      pos_desired[0][2] = -0.025;
      pos_desired[1][2] = -0.025;
      pos_desired[2][2] = -0.025;
      pos_desired[3][2] = -0.025;
  }

  Eigen::MatrixXd Lambda_swingFoot_LF_ = (j_.topRows(3) * M_bar_.inverse() * Projection_matrix_ * (j_.topRows(3)).transpose()).inverse();
  Eigen::MatrixXd Lambda_swingFoot_RF_ = (j_.block(3, 0, 3, info_.generalizedCoordinatesNum) * M_bar_.inverse() * Projection_matrix_ * (j_.block(3, 0, 3, info_.generalizedCoordinatesNum)).transpose()).inverse();
  Eigen::MatrixXd Lambda_swingFoot_LH_ = (j_.block(6, 0, 3, info_.generalizedCoordinatesNum) * M_bar_.inverse() * Projection_matrix_ * (j_.block(6, 0, 3, info_.generalizedCoordinatesNum)).transpose()).inverse();
  Eigen::MatrixXd Lambda_swingFoot_RH_ = (j_.block(9, 0, 3, info_.generalizedCoordinatesNum) * M_bar_.inverse() * Projection_matrix_ * (j_.block(9, 0, 3, info_.generalizedCoordinatesNum)).transpose()).inverse();
  // Eigen::MatrixXd Lambda_swingFoot_LF_ = getSVDBasedInverse(j_.topRows(3) * M_bar_.inverse() * Projection_matrix_ * (j_.topRows(3)).transpose());
  // Eigen::MatrixXd Lambda_swingFoot_RF_ = getSVDBasedInverse(j_.block(3, 0, 3, info_.generalizedCoordinatesNum) * M_bar_.inverse() * Projection_matrix_ * (j_.block(3, 0, 3, info_.generalizedCoordinatesNum)).transpose());
  // Eigen::MatrixXd Lambda_swingFoot_LH_ = getSVDBasedInverse(j_.block(6, 0, 3, info_.generalizedCoordinatesNum) * M_bar_.inverse() * Projection_matrix_ * (j_.block(6, 0, 3, info_.generalizedCoordinatesNum)).transpose());
  // Eigen::MatrixXd Lambda_swingFoot_RH_ = getSVDBasedInverse(j_.block(9, 0, 3, info_.generalizedCoordinatesNum) * M_bar_.inverse() * Projection_matrix_ * (j_.block(9, 0, 3, info_.generalizedCoordinatesNum)).transpose());

  Eigen::MatrixXd inv_Lambda_swingFoot_LF_ = j_.topRows(3) * M_bar_.inverse() * Projection_matrix_ * (j_.topRows(3)).transpose();
  Eigen::MatrixXd inv_Lambda_swingFoot_RF_ = j_.block(3, 0, 3, info_.generalizedCoordinatesNum) * M_bar_.inverse() * Projection_matrix_ * (j_.block(3, 0, 3, info_.generalizedCoordinatesNum)).transpose();
  Eigen::MatrixXd inv_Lambda_swingFoot_LH_ = j_.block(6, 0, 3, info_.generalizedCoordinatesNum) * M_bar_.inverse() * Projection_matrix_ * (j_.block(6, 0, 3, info_.generalizedCoordinatesNum)).transpose();
  Eigen::MatrixXd inv_Lambda_swingFoot_RH_ = j_.block(9, 0, 3, info_.generalizedCoordinatesNum) * M_bar_.inverse() * Projection_matrix_ * (j_.block(9, 0, 3, info_.generalizedCoordinatesNum)).transpose();

}

}  // namespace legged
