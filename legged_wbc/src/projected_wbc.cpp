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

  friction_coef_ = 0.8; // TODO (@Shengzhi): hardcoded coeeficient

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
  torque_command_.setZero(info_.generalizedCoordinatesNum);
  v_mrt_prev_.setZero(info_.generalizedCoordinatesNum);
  F_base_ext_est_.setZero(6);
  Hessian_matrix_.setZero(info_.actuatedDofNum, info_.actuatedDofNum);
  f_vector_.setZero(info_.actuatedDofNum);
  Constraint_matrix_.setZero(info_.actuatedDofNum, info_.actuatedDofNum); 
  lowerBound_.setZero(info_.actuatedDofNum);
  upperBound_.setZero(info_.actuatedDofNum);
  Selection_matrix_.setZero(info_.generalizedCoordinatesNum, info_.actuatedDofNum);
  Selection_matrix_.block<12,12>(6,0) = Eigen::MatrixXd::Identity(12, 12);
  Selection_matrix_support_.setZero(info_.generalizedCoordinatesNum, info_.actuatedDofNum);
  Selection_matrix_support_.block<12,12>(6,0) = Eigen::MatrixXd::Identity(12, 12);    // TODO: Hardcoded for test
  Selection_matrix_support_(6, 0) = 0, Selection_matrix_support_(7, 1) = 0, Selection_matrix_support_(8, 2) = 0;  // TODO: Hardcoded for test
  Selection_matrix_force_.setZero(info_.generalizedCoordinatesNum, info_.actuatedDofNum);
  Selection_matrix_force_(6, 0) = 1, Selection_matrix_force_(7, 1) = 1, Selection_matrix_force_(8, 2) = 1;    // TODO: Hardcoded for test
  TorqueLowerBound_.setZero(info_.actuatedDofNum);
  TorqueLowerBound_ << -torque_limits_, -torque_limits_, -torque_limits_, -torque_limits_;
  TorqueUpperBound_.setZero(info_.actuatedDofNum);
  TorqueUpperBound_ << torque_limits_, torque_limits_, torque_limits_, torque_limits_;

  count_ = 0;
  waitTime_ = 5000;
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

  // Get foot friction constraint
  getFootFrictionConeConstraint();

  // Get the manipulability of the Jacobian_contact_
  getManipulability(Jacobian_contact_);

  // Calculate Robot Dynamics
  calculateRobotDynamics(data);

  // Calculate Reference Torque
  calculateReferenceTorque(mode);

  // Get QP soft and hard constraints
  getQPSoftAndHardConstraint_noArm();

  // Get the Hierarchical QP solution
  vector_t Solution = getHQPSolution();

  ++count_;
  return Solution;

  // // Optimization problem
  // Task task_0 = formulateFloatingBaseEomTask() + formulateTorqueLimitsTask() + formulateFrictionConeTask() +
  //               formulateNoContactMotionTask();
  // Task task_1 = formulateBaseAccelTask() + formulateSwingLegTask();
  // Task task_2 = formulateContactForceTask();
  // HoQp ho_qp(task_2, std::make_shared<HoQp>(task_1, std::make_shared<HoQp>(task_0)));

  // return ho_qp.getSolutions();
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
  contact_num_for_support_ = 0;
  contact_num_for_support_ = contact_flag_[1] + contact_flag_[2] + contact_flag_[3];
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
  contact_num_for_force_control_ = 0;
  contact_num_for_force_control_ = contact_flag_[0];
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

void ProjectedWBC::getFootFrictionConeConstraint()
{
  // x, y, and z axis direction of the contact surface. TODO: @ Shengzhi: this can be designed in the future. From now, we set them as constant.
  Eigen::Matrix<double, 1, 3> n_x_, n_y_, n_z_;
  n_x_ << 1, 0, 0;
  n_y_ << 0, 1, 0;
  n_z_ << 0, 0, 1;

  // Lower and upper bound for each foot
  Eigen::Matrix<double, 5, 1> low_d_i_, high_d_i_;
  low_d_i_ << -1e8, -1e8, 0.0, 0.0, 0.0;
  high_d_i_ << 0.0, 0.0, 1e8, 1e8, 1e8;

  int FrictionConstraint_count_ = 0;
  if (num_contacts_ != 0){ 
      for (int i = 0; i < 4; i++){
          if (contact_flag_[i]){
              switch (i){
                  case 0:         // LF foot
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_, 3 * FrictionConstraint_count_) = n_x_ - friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 1, 3 * FrictionConstraint_count_) = n_y_ - friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 2, 3 * FrictionConstraint_count_) = n_x_ + friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 3, 3 * FrictionConstraint_count_) = n_y_ + friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 4, 3 * FrictionConstraint_count_) = n_z_;

                    low_d_vector_.segment<5>(5 * FrictionConstraint_count_) = low_d_i_;
                    high_d_vector_.segment<5>(5 * FrictionConstraint_count_) = high_d_i_;                             
                  
                    FrictionConstraint_count_++;
                    break;
                  case 1:         // RF foot
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_, 3 * FrictionConstraint_count_) = n_x_ - friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 1, 3 * FrictionConstraint_count_) = n_y_ - friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 2, 3 * FrictionConstraint_count_) = n_x_ + friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 3, 3 * FrictionConstraint_count_) = n_y_ + friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 4, 3 * FrictionConstraint_count_) = n_z_;

                    low_d_vector_.segment<5>(5 * FrictionConstraint_count_) = low_d_i_;
                    high_d_vector_.segment<5>(5 * FrictionConstraint_count_) = high_d_i_;

                    FrictionConstraint_count_++;
                    break;  
                  case 2:         // LH foot
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_, 3 * FrictionConstraint_count_) = n_x_ - friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 1, 3 * FrictionConstraint_count_) = n_y_ - friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 2, 3 * FrictionConstraint_count_) = n_x_ + friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 3, 3 * FrictionConstraint_count_) = n_y_ + friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 4, 3 * FrictionConstraint_count_) = n_z_;

                    low_d_vector_.segment<5>(5 * FrictionConstraint_count_) = low_d_i_;
                    high_d_vector_.segment<5>(5 * FrictionConstraint_count_) = high_d_i_;

                    FrictionConstraint_count_++;
                    break; 
                  case 3:         // RH foot
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_, 3 * FrictionConstraint_count_) = n_x_ - friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 1, 3 * FrictionConstraint_count_) = n_y_ - friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 2, 3 * FrictionConstraint_count_) = n_x_ + friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 3, 3 * FrictionConstraint_count_) = n_y_ + friction_coef_ * n_z_;
                    FrictionConstraint_matrix_.block<1, 3>(5 * FrictionConstraint_count_ + 4, 3 * FrictionConstraint_count_) = n_z_;

                    low_d_vector_.segment<5>(5 * FrictionConstraint_count_) = low_d_i_;
                    high_d_vector_.segment<5>(5 * FrictionConstraint_count_) = high_d_i_;

                    FrictionConstraint_count_++;
                    break;                      
              }
          }
      }
  }

  // Cs: Friction cone constraint for the foot supporting the torso exclusively (TODO: hardcoded)
  int FrictionConstraint_support_count_ = 0;
  if (contact_num_for_support_ != 0){ 
      for (int i = 1; i < 4; i++){
          if (contact_flag_[i]){
              switch (i){
                  // case 0:         // LF foot
                  //     for(auto& contact_: anymal.back()->getContacts()) {
                  //         if (contact_.skip()) continue; /// if the contact is internal, one contact point is set to 'skip'
                  //         if (footIndices_[0] == contact_.getlocalBodyIndex()) {
                  //             auto n_x_ = contact_.getContactFrame().e().row(0);
                  //             auto n_y_ = contact_.getContactFrame().e().row(1);
                  //             auto n_z_ = contact_.getNormal().e().transpose();
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_, 3 * FrictionConstraint_support_count_) = n_x_ - friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 1, 3 * FrictionConstraint_support_count_) = n_y_ - friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 2, 3 * FrictionConstraint_support_count_) = n_x_ + friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 3, 3 * FrictionConstraint_support_count_) = n_y_ + friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 4, 3 * FrictionConstraint_support_count_) = n_z_;

                  //             low_d_vector_.segment<5>(5 * FrictionConstraint_support_count_) = low_d_i_;
                  //             high_d_vector_.segment<5>(5 * FrictionConstraint_support_count_) = high_d_i_;                              
                  //         }
                  //     }
                  //     FrictionConstraint_support_count_++;
                  //     break;
                  case 1:         // RF foot
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_, 3 * FrictionConstraint_support_count_) = n_x_ - friction_coef_ * n_z_;
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 1, 3 * FrictionConstraint_support_count_) = n_y_ - friction_coef_ * n_z_;
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 2, 3 * FrictionConstraint_support_count_) = n_x_ + friction_coef_ * n_z_;
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 3, 3 * FrictionConstraint_support_count_) = n_y_ + friction_coef_ * n_z_;
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 4, 3 * FrictionConstraint_support_count_) = n_z_;

                    low_D_m_vector_.segment<5>(5 * FrictionConstraint_support_count_) = low_d_i_;
                    high_D_m_vector_.segment<5>(5 * FrictionConstraint_support_count_) = high_d_i_;

                    FrictionConstraint_support_count_++;
                    break;  
                  case 2:         // LH foot
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_, 3 * FrictionConstraint_support_count_) = n_x_ - friction_coef_ * n_z_;
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 1, 3 * FrictionConstraint_support_count_) = n_y_ - friction_coef_ * n_z_;
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 2, 3 * FrictionConstraint_support_count_) = n_x_ + friction_coef_ * n_z_;
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 3, 3 * FrictionConstraint_support_count_) = n_y_ + friction_coef_ * n_z_;
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 4, 3 * FrictionConstraint_support_count_) = n_z_;

                    low_D_m_vector_.segment<5>(5 * FrictionConstraint_support_count_) = low_d_i_;
                    high_D_m_vector_.segment<5>(5 * FrictionConstraint_support_count_) = high_d_i_;

                    FrictionConstraint_support_count_++;
                    break; 
                  case 3:         // RH foot
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_, 3 * FrictionConstraint_support_count_) = n_x_ - friction_coef_ * n_z_;
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 1, 3 * FrictionConstraint_support_count_) = n_y_ - friction_coef_ * n_z_;
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 2, 3 * FrictionConstraint_support_count_) = n_x_ + friction_coef_ * n_z_;
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 3, 3 * FrictionConstraint_support_count_) = n_y_ + friction_coef_ * n_z_;
                    FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 4, 3 * FrictionConstraint_support_count_) = n_z_;

                    low_D_m_vector_.segment<5>(5 * FrictionConstraint_support_count_) = low_d_i_;
                    high_D_m_vector_.segment<5>(5 * FrictionConstraint_support_count_) = high_d_i_;

                    FrictionConstraint_support_count_++;
                    break;                      
              }
          }
      }
  }

  // Cf: Friction cone constraint for the foot using force control exclusively (TODO: hardcoded)
  int FrictionConstraint_force_control_count_ = 0;
  if (contact_num_for_force_control_ != 0){ 
      for (int i = 0; i < 1; i++){
          if (contact_flag_[i]){
              switch (i){
                  case 0:         // LF foot
                    FrictionConstraint_for_force_control_matrix_.block<1, 3>(5 * FrictionConstraint_force_control_count_, 3 * FrictionConstraint_force_control_count_) = n_x_ - friction_coef_ * n_z_;
                    FrictionConstraint_for_force_control_matrix_.block<1, 3>(5 * FrictionConstraint_force_control_count_ + 1, 3 * FrictionConstraint_force_control_count_) = n_y_ - friction_coef_ * n_z_;
                    FrictionConstraint_for_force_control_matrix_.block<1, 3>(5 * FrictionConstraint_force_control_count_ + 2, 3 * FrictionConstraint_force_control_count_) = n_x_ + friction_coef_ * n_z_;
                    FrictionConstraint_for_force_control_matrix_.block<1, 3>(5 * FrictionConstraint_force_control_count_ + 3, 3 * FrictionConstraint_force_control_count_) = n_y_ + friction_coef_ * n_z_;
                    FrictionConstraint_for_force_control_matrix_.block<1, 3>(5 * FrictionConstraint_force_control_count_ + 4, 3 * FrictionConstraint_force_control_count_) = n_z_;

                    low_D_f_vector_.segment<5>(5 * FrictionConstraint_force_control_count_) = low_d_i_;
                    high_D_f_vector_.segment<5>(5 * FrictionConstraint_force_control_count_) = high_d_i_;                              

                    FrictionConstraint_force_control_count_++;
                    break;
                  // case 1:         // RF foot
                  //     for(auto& contact_: anymal.back()->getContacts()) {
                  //         if (contact_.skip()) continue; /// if the contact is internal, one contact point is set to 'skip'
                  //         if (footIndices_[1] == contact_.getlocalBodyIndex()) {
                  //             auto n_x_ = contact_.getContactFrame().e().row(0);
                  //             auto n_y_ = contact_.getContactFrame().e().row(1);
                  //             auto n_z_ = contact_.getNormal().e().transpose();
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_, 3 * FrictionConstraint_support_count_) = n_x_ - friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 1, 3 * FrictionConstraint_support_count_) = n_y_ - friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 2, 3 * FrictionConstraint_support_count_) = n_x_ + friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 3, 3 * FrictionConstraint_support_count_) = n_y_ + friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 4, 3 * FrictionConstraint_support_count_) = n_z_;

                  //             low_D_m_vector_.segment<5>(5 * FrictionConstraint_support_count_) = low_d_i_;
                  //             high_D_m_vector_.segment<5>(5 * FrictionConstraint_support_count_) = high_d_i_;
                  //         }
                  //     }
                  //     FrictionConstraint_support_count_++;
                  //     break;  
                  // case 2:         // LH foot
                  //     for(auto& contact_: anymal.back()->getContacts()) {
                  //         if (contact_.skip()) continue; /// if the contact is internal, one contact point is set to 'skip'
                  //         if (footIndices_[2] == contact_.getlocalBodyIndex()) {
                  //             auto n_x_ = contact_.getContactFrame().e().row(0);
                  //             auto n_y_ = contact_.getContactFrame().e().row(1);
                  //             auto n_z_ = contact_.getNormal().e().transpose();
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_, 3 * FrictionConstraint_support_count_) = n_x_ - friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 1, 3 * FrictionConstraint_support_count_) = n_y_ - friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 2, 3 * FrictionConstraint_support_count_) = n_x_ + friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 3, 3 * FrictionConstraint_support_count_) = n_y_ + friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 4, 3 * FrictionConstraint_support_count_) = n_z_;

                  //             low_D_m_vector_.segment<5>(5 * FrictionConstraint_support_count_) = low_d_i_;
                  //             high_D_m_vector_.segment<5>(5 * FrictionConstraint_support_count_) = high_d_i_;
                  //         }
                  //     }
                  //     FrictionConstraint_support_count_++;
                  //     break; 
                  // case 3:         // RH foot
                  //     for(auto& contact_: anymal.back()->getContacts()) {
                  //         if (contact_.skip()) continue; /// if the contact is internal, one contact point is set to 'skip'
                  //         if (footIndices_[3] == contact_.getlocalBodyIndex()) {
                  //             auto n_x_ = contact_.getContactFrame().e().row(0);
                  //             auto n_y_ = contact_.getContactFrame().e().row(1);
                  //             auto n_z_ = contact_.getNormal().e().transpose();
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_, 3 * FrictionConstraint_support_count_) = n_x_ - friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 1, 3 * FrictionConstraint_support_count_) = n_y_ - friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 2, 3 * FrictionConstraint_support_count_) = n_x_ + friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 3, 3 * FrictionConstraint_support_count_) = n_y_ + friction_coef_ * n_z_;
                  //             FrictionConstraint_for_support_matrix_.block<1, 3>(5 * FrictionConstraint_support_count_ + 4, 3 * FrictionConstraint_support_count_) = n_z_;

                  //             low_D_m_vector_.segment<5>(5 * FrictionConstraint_support_count_) = low_d_i_;
                  //             high_D_m_vector_.segment<5>(5 * FrictionConstraint_support_count_) = high_d_i_;
                  //         }
                  //     }
                  //     FrictionConstraint_support_count_++;
                  //     break;                      
              }
          }
      }
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

  // Using direct inversion  
  Eigen::VectorXd swingFoot_force_command_LF_ = Lambda_swingFoot_LF_ * acc_desired[0] 
                          + task_D_gain_foot * (vel_desired[0] - vel_measured[0]) 
                          + task_P_gain_foot * (pos_desired[0] - pos_measured[0]) 
                          + Lambda_swingFoot_LF_ * (j_.topRows(3) * M_bar_.inverse() * h_bar_ 
                          - dj_.topRows(3) * measured_v_);
  Eigen::VectorXd swingFoot_force_command_RF_ = Lambda_swingFoot_RF_ * acc_desired[1] 
                          + task_D_gain_foot * (vel_desired[1] - vel_measured[1]) 
                          + task_P_gain_foot * (pos_desired[1] - pos_measured[1]) 
                          + Lambda_swingFoot_RF_ * (j_.block(3, 0, 3, info_.generalizedCoordinatesNum) * M_bar_.inverse() * h_bar_ 
                          - dj_.block(3, 0, 3, info_.generalizedCoordinatesNum) * measured_v_);
  Eigen::VectorXd swingFoot_force_command_LH_ = Lambda_swingFoot_LH_ * acc_desired[2] 
                          + task_D_gain_foot * (vel_desired[2] - vel_measured[2]) 
                          + task_P_gain_foot * (pos_desired[2] - pos_measured[2]) 
                          + Lambda_swingFoot_LH_ * (j_.block(6, 0, 3, info_.generalizedCoordinatesNum) * M_bar_.inverse() * h_bar_ 
                          - dj_.block(6, 0, 3, info_.generalizedCoordinatesNum) * measured_v_);
  Eigen::VectorXd swingFoot_force_command_RH_ = Lambda_swingFoot_RH_ * acc_desired[3] 
                          + task_D_gain_foot * (vel_desired[3] - vel_measured[3]) 
                          + task_P_gain_foot * (pos_desired[3] - pos_measured[3]) 
                          + Lambda_swingFoot_RH_ * (j_.block(9, 0, 3, info_.generalizedCoordinatesNum) * M_bar_.inverse() * h_bar_ 
                          - dj_.block(9, 0, 3, info_.generalizedCoordinatesNum) * measured_v_);

  // if (count_ < 2 * waitTime_){
    // Eigen::Vector3d desired_base_position_;
    // Eigen::Vector3d desired_base_orientation_euler_;
    // desired_base_position_ << 0.0, 0.0, 0.57;
    // desired_base_orientation_euler_ << 0.0, 0.0, 0.0;
  // }
  Eigen::VectorXd posture_error_(6);
  Eigen::VectorXd posture_error_compensation_(6);
  posture_error_.head(3) = basePosition_desired - measured_q_.head<3>();
  // posture_error_compensation_.head(3) = desired_base_position_compensation_ - measured_q_.head<3>();

  // Eigen::Matrix3d desired_base_rotation_matrix_ = ConventionRules::Quaternion2RotationMatrix(desired_base_orientation_quat_);
  // ConventionRules::GetBaseOrientationError(I_R_Base_current_, desired_base_rotation_matrix_, orientation_error_);
  posture_error_.tail(3) = baseOrientation_desired - measured_q_.segment<3>(3);
  // posture_error_compensation_.tail(3) = orientation_error_;
  Eigen::VectorXd velocity_error_ = v_desired.head(6) - measured_v_.head(6); 
  auto idx_base_ = model.getFrameId("base");

  // // torque command assuming all joints are active
  // if (count_ == 1.8 * waitTime_ + 0){ 
  //     F_compensation_ = Lambda_ * desired_base_classic_acceleration_global_ + task_D_gain_base_ * velocity_error_ + task_P_gain_base_ * posture_error_;   // Using direct inversion 
  //     // F_compensation_ = inv_Lambda_.colPivHouseholderQr().solve(desired_base_classic_acceleration_global_) + task_D_gain_base_ * velocity_error_ + task_P_gain_base_ * posture_error_compensation_;   // Solving linear least squares systems, instead of using direct inversion
  // }       
  // F_compensation_.setZero();
  Eigen::VectorXd base_force_command_ = Lambda_ * qbaseDdot_ + task_D_gain_base_ * velocity_error_ + task_P_gain_base_ * posture_error_ 
                                  + Lambda_ * (Jacobian_base_ * M_bar_.inverse() * h_bar_ - d_Jacobian_base_ * measured_v_)
                                  ; // + F_compensation_;  // Using direct inversion 
  // Eigen::VectorXd base_force_command_ = inv_Lambda_.colPivHouseholderQr().solve(desired_base_classic_acceleration_global_)
  //                                 + task_D_gain_base_ * velocity_error_ + task_P_gain_base_ * posture_error_ 
  //                                 + inv_Lambda_.colPivHouseholderQr().solve((Jacobian_base_ * M_bar_.inverse() * h_bar_ - d_Jacobian_base_ * anymal_dq_mrt_pinocchio_))
  //                                 + F_compensation_;  // Solving linear least squares systems, instead of using direct inversion 
  Eigen::VectorXd base_torque_command_ = Jacobian_base_.transpose() * base_force_command_;
  if (num_contacts_ == 0){
      // auto g_vector_ = pinocchio::computeGeneralizedGravity(model_, data_, anymal_q_mrt_pinocchio_);
      // Eigen::VectorXd base_torque_command_ = g_vector_;
      base_torque_command_ = Eigen::MatrixXd::Zero(info_.generalizedCoordinatesNum, 1);
  }

  Eigen::VectorXd swingLeg_torque_command_ = Eigen::MatrixXd::Zero(info_.generalizedCoordinatesNum, 1);   // TODO (@Shengzhi): hardcoded foot tracking torque
  torque_command_.setZero();
  torque_command_ = Projection_matrix_ * (base_torque_command_ + swingLeg_torque_command_);      // Our method
  // torque_command_ = pseudoInverse(Projection_matrix_ * Selection_matrix_full_, 0.02) * (base_torque_command_ + swingLeg_torque_command_);      // Guiyang's method

//   Eigen::VectorXd anymal_ddq_mrt_pinocchio_ = (measured_v_ - v_mrt_prev_) / dt;
//   v_mrt_prev_ = measured_v_;

//   Eigen::VectorXd F_base_ext_est_ = inv_Lambda_.colPivHouseholderQr().solve((anymal_ddq_mrt_pinocchio_.head(6) - qbaseDdot_))
//                     - task_D_gain_base_ * velocity_error_ - task_P_gain_base_ * posture_error_;   // Solving linear least squares systems, instead of using direct inversion
// // std::cerr << "F_base_ext_est_: \n" << F_base_ext_est_ << "\n";    
//   F_base_ext_est_.setZero();
// //     F_base_ext_est_[2] = 10;

}

void ProjectedWBC::getQPSoftAndHardConstraint_noArm()
{
  // Constraint_matrix_.resize(info_.actuatedDofNum + 5 * num_contacts_, info_.actuatedDofNum); 
  // lowerBound_.resize(info_.actuatedDofNum + 5 * num_contacts_);
  // upperBound_.resize(info_.actuatedDofNum + 5 * num_contacts_);
  // Constraint_matrix_.resize(info_.actuatedDofNum + 5 * contact_num_for_support_, info_.actuatedDofNum); 
  // lowerBound_.resize(info_.actuatedDofNum + 5 * contact_num_for_support_);
  // upperBound_.resize(info_.actuatedDofNum + 5 * contact_num_for_support_);
  Hessian_matrix_.setZero(); f_vector_.setZero(); Constraint_matrix_.setZero();
  lowerBound_.setZero(); upperBound_.setZero();

  // First P, then I - P
  Hessian_matrix_ = 2 * Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_;       // Our method
  // Hessian_matrix_ = 2 * Selection_matrix_.transpose() * Projection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_;       // Guiyang's method
  // Hessian_matrix_ = 2 * Selection_matrix_.transpose() * (Projection_matrix_ + Projection_matrix_.transpose() * (M_bar_.inverse()).transpose() * Jacobian_contact_.transpose() * Jacobian_contact_ * M_bar_.inverse() * Projection_matrix_) * Selection_matrix_;
  // Hessian_matrix_ = 2 * Selection_matrix_.transpose() * (Projection_matrix_.transpose() * (M_bar_.inverse()).transpose() * Jacobian_contact_.transpose() * Jacobian_contact_ * M_bar_.inverse() * Projection_matrix_) * Selection_matrix_;
  // Hessian_matrix_ = 2 * Selection_matrix_.transpose() * (Inertial_matrix_.inverse()).transpose() * Jacobian_contact_.transpose() * Jacobian_contact_ * Inertial_matrix_.inverse() * Selection_matrix_;
  // Hessian_matrix_ = 2 * Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_ + 2 * Selection_matrix_.transpose() * (Inertial_matrix_.inverse()).transpose() * Jacobian_contact_.transpose() * Jacobian_contact_ * Inertial_matrix_.inverse() * Selection_matrix_;
  // Hessian_matrix_ = 2 * Selection_matrix_.transpose() * (Eigen::MatrixXd::Identity(18, 18) - Projection_matrix_) * Selection_matrix_;
  Hessian_matrix_ = (Hessian_matrix_ + Hessian_matrix_.transpose())/2;

  f_vector_ = -2 * Selection_matrix_support_.transpose() * Projection_matrix_.transpose() * torque_command_;
  // f_vector_ = -2 * Selection_matrix_.transpose() * (Eigen::MatrixXd::Identity(18, 18) - Projection_matrix_).transpose() * torque_command_;
  // f_vector_ = -2 * Selection_matrix_.transpose() * Projection_matrix_.transpose() * torque_command_ - 2 * (Selection_matrix_.transpose() * Projection_matrix_.transpose() * (M_bar_.inverse()).transpose() * Jacobian_contact_.transpose()) * (Jacobian_contact_ * M_bar_.inverse() * Projection_matrix_ * h_vector_ - Jacobian_contact_ * M_bar_.inverse() * d_projection_matrix_ * anymal_dq_mrt_pinocchio_ - d_Jacobian_contact_ * anymal_dq_mrt_pinocchio_);
  // f_vector_ = -2 * (Selection_matrix_.transpose() * Projection_matrix_.transpose() * (M_bar_.inverse()).transpose() * Jacobian_contact_.transpose()) * (Jacobian_contact_ * M_bar_.inverse() * Projection_matrix_ * h_vector_ - Jacobian_contact_ * M_bar_.inverse() * d_projection_matrix_ * anymal_dq_mrt_pinocchio_ - d_Jacobian_contact_ * anymal_dq_mrt_pinocchio_);
  // f_vector_ = 2 * (Selection_matrix_.transpose() * (Inertial_matrix_.inverse()).transpose() * Jacobian_contact_.transpose()) * (Jacobian_contact_ * Inertial_matrix_.inverse() * Jacobian_contact_.transpose() * footContactForce_contactedOnly_ - Jacobian_contact_ * Inertial_matrix_.inverse() * h_vector_ + d_Jacobian_contact_ * anymal_dq_mrt_pinocchio_);
  // f_vector_ = -2 * Selection_matrix_.transpose() * Projection_matrix_.transpose() * torque_command_ + 2 * (Selection_matrix_.transpose() * (Inertial_matrix_.inverse()).transpose() * Jacobian_contact_.transpose()) * (Jacobian_contact_ * Inertial_matrix_.inverse() * Jacobian_contact_.transpose() * footContactForce_contactedOnly_ - Jacobian_contact_ * Inertial_matrix_.inverse() * h_vector_ + d_Jacobian_contact_ * anymal_dq_mrt_pinocchio_);
  // f_vector_ = 2 * Selection_matrix_.transpose() * Projection_matrix_.transpose() * (Selection_matrix_ * torque_PD - torque_command_);

  // Friction cone constraint, torque constraint, and unilateral constraint.
  if (contact_num_for_support_ != 0){
      Eigen::MatrixXd pinv_Jacobian_contact_transpose_ = pseudoInverse(Jacobian_contact_.transpose());   //J_c_T_pinv
      // if (manipulability_ < 0.001){
      //     pinv_Jacobian_contact_transpose_.setZero();
      //     pinv_Jacobian_contact_transpose_ = Jacobian_contact_ * (Jacobian_contact_.transpose() * Jacobian_contact_ 
      //                                         + std::pow(damped_lambda_, 2) * Eigen::MatrixXd::Identity(Jacobian_contact_.cols(), Jacobian_contact_.cols())).inverse();  //J_c_T_pinv (damped solution)
      // }
      // Eigen::MatrixXd W_matrix_ = -FrictionConstraint_matrix_ * pinv_Jacobian_contact_transpose_ * (Eigen::MatrixXd::Identity(18, 18) - Projection_matrix_) 
      //                 * (Eigen::MatrixXd::Identity(18, 18) - Inertial_matrix_ * M_bar_.inverse() * Projection_matrix_) * Selection_matrix_;    // W

      // Eigen::VectorXd rho_matrix_ = pinv_Jacobian_contact_transpose_ * (Eigen::MatrixXd::Identity(18, 18) - Projection_matrix_)
      //                     * ((Eigen::MatrixXd::Identity(18, 18) - Inertial_matrix_ * M_bar_.inverse() * Projection_matrix_) * h_vector_ 
      //                     + Inertial_matrix_ * M_bar_.inverse() * d_projection_matrix_ * anymal_dq_mrt_pinocchio_);     // rho (Using direct inversion)
      // Eigen::VectorXd rho_matrix_ = Jacobian_contact_.transpose().bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(((Eigen::MatrixXd::Identity(18, 18) - Projection_matrix_)
      //                     * ((Eigen::MatrixXd::Identity(18, 18) - Inertial_matrix_ * M_bar_.inverse() * Projection_matrix_) * h_vector_ 
      //                     + Inertial_matrix_ * M_bar_.inverse() * d_projection_matrix_ * anymal_dq_mrt_pinocchio_)));     // rho (Solving linear least squares systems, instead of using direct inversion)

      Eigen::MatrixXd W_s_matrix = FrictionConstraint_for_support_matrix_ * pseudoInverse(Jacobian_contact_support_.transpose()) 
                                    * (Eigen::MatrixXd::Identity(info_.generalizedCoordinatesNum, info_.generalizedCoordinatesNum) - Projection_matrix_)
                                    * (Inertial_matrix_ * M_bar_.inverse() * Projection_matrix_ - Eigen::MatrixXd::Identity(info_.generalizedCoordinatesNum, info_.generalizedCoordinatesNum)) * Selection_matrix_support_;
      Eigen::VectorXd rho_s_matrix_ = FrictionConstraint_for_support_matrix_ * pseudoInverse(Jacobian_contact_support_.transpose()) * (Eigen::MatrixXd::Identity(info_.generalizedCoordinatesNum, info_.generalizedCoordinatesNum) - Projection_matrix_) 
                                      * ((Inertial_matrix_ * M_bar_.inverse() * (-Projection_matrix_ * h_vector_ + d_projection_matrix_ * measured_v_) + h_vector_) 
                                      + (Inertial_matrix_ * M_bar_.inverse() * Projection_matrix_ - Eigen::MatrixXd::Identity(info_.generalizedCoordinatesNum, info_.generalizedCoordinatesNum)) );
                                      // * (Jacobian_base_.transpose() * F_base_ext_est_));

      // Constraint_matrix_.topRows(12) = Eigen::MatrixXd::Identity(12, 12);    // Torque limit not theoratical correct
      Constraint_matrix_.topRows(12) = Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_;    // Torque limit theoratical correct
      // Constraint_matrix_.topRows(12) = Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_;    // Torque limit theoratical correct
      // Constraint_matrix_.bottomRows(5 * contact_num_for_support_) = W_s_matrix;
      
      lowerBound_.head(12) = TorqueLowerBound_;
      // lowerBound_.head(12) = TorqueLowerBound_unlimited_;
      // lowerBound_.tail(5 * contact_num_for_support_) = low_D_m_vector_ - rho_s_matrix_;
      // lowerBound_.head(12) = TorqueLowerBound_ - torque_PD;
      // lowerBound_.tail(5 * num_contacts_) = low_d_vector_ - FrictionConstraint_matrix_ * rho_matrix_ - W_matrix_ * torque_PD;
      
      upperBound_.head(12) = TorqueUpperBound_;
      // upperBound_.head(12) = TorqueUpperBound_unlimited_; 
      // upperBound_.tail(5 * contact_num_for_support_) = high_D_m_vector_ - rho_s_matrix_;   
      // upperBound_.head(12) = TorqueUpperBound_ - torque_PD;          
      // upperBound_.tail(5 * num_contacts_) = high_d_vector_ - FrictionConstraint_matrix_ * rho_matrix_ - W_matrix_ * torque_PD;    
  }
  else{
      // Constraint_matrix_ = Eigen::MatrixXd::Identity(12, 12);    // Torque limit not theoratical correct
      Constraint_matrix_ = Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_;    // Torque limit theoratical correct
      // Constraint_matrix_ = Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_;    // Torque limit theoratical correct
      lowerBound_ = TorqueLowerBound_;
      upperBound_ = TorqueUpperBound_;
      // lowerBound_ = TorqueLowerBound_unlimited_;
      // upperBound_ = TorqueUpperBound_unlimited_;
      // lowerBound_ = TorqueLowerBound_ - torque_PD;
      // upperBound_ = TorqueUpperBound_ - torque_PD;            
  } 
}

vector_t ProjectedWBC::getHQPSolution()
{
  unsigned int numOfVar = Hessian_matrix_.rows();
  unsigned int numOfRows = Constraint_matrix_.rows(); 

  Eigen::SparseMatrix<double> hessian(info_.actuatedDofNum, info_.actuatedDofNum);
  for(int i=0; i<numOfVar; i++){          
      for(int j=0; j<numOfVar; j++){     
          hessian.insert(i, j) = Hessian_matrix_(i, j);
      }
  }

  Eigen::SparseMatrix<double> linearMatrix(numOfRows, info_.actuatedDofNum);
  for(int i=0; i<numOfRows; i++){         
      for(int j=0; j<numOfVar; j++){      
          linearMatrix.insert(i, j) = Constraint_matrix_(i, j); 
      }
  }

  Eigen::VectorXd gradient(info_.actuatedDofNum);
  gradient = f_vector_;

  // Update the QP solver
  solver.clearSolver();
  solver.data()->clearHessianMatrix();
  solver.data()->clearLinearConstraintsMatrix();

  solver.settings()->setVerbosity(false);
  solver.settings()->setAlpha(1.0);
  solver.settings()->setWarmStart(true);

  solver.data()->setNumberOfVariables(info_.actuatedDofNum);
  solver.data()->setNumberOfConstraints(numOfRows);

  if(!solver.data()->setHessianMatrix(hessian)) return vector_t();
  if(!solver.data()->setGradient(gradient)) return vector_t();
  if(!solver.data()->setLinearConstraintsMatrix(linearMatrix)) return vector_t();
  if(!solver.data()->setLowerBound(lowerBound_)) return vector_t(); 
  if(!solver.data()->setUpperBound(upperBound_)) return vector_t(); 

  if(!solver.initSolver()) return vector_t();

  solver.solve();

  Eigen::VectorXd QPSolution = solver.getSolution();

  // Second hierachical QP (first in P, then in I - P)
  Eigen::MatrixXd Identity_matrix = Eigen::MatrixXd::Identity(info_.generalizedCoordinatesNum, info_.generalizedCoordinatesNum);

  // Used for the sinwave force tracking
  Eigen::VectorXd desired_contact_force_(3);
  desired_contact_force_ << 0, 0, 140; // TODO: Hardcoded

  // Eigen::VectorXd tau_tot_des = (Identity_matrix - Projection_matrix_)
  //                                 * (Inertial_matrix_ * M_bar_.inverse() * (torque_command_ 
  //                                 - Projection_matrix_ * h_vector_ + d_projection_matrix_ * anymal_dq_mrt_pinocchio_ + Projection_matrix_ * Jacobian_base_.transpose() * F_base_ext_est_) + h_vector_)
  //                                 - (Identity_matrix - Projection_matrix_) * Jacobian_base_.transpose() * F_base_ext_est_;
  // Eigen::VectorXd desired_contact_force_ = pseudoInverse(Jacobian_contact_.transpose()) * tau_tot_des;
  // if (count_ == 2 * waitTime_ + 2000){
  //     desired_init_contact_force = desired_contact_force_;
  // }
  if (count_ > 2 * waitTime_ + 2000){
  // if (count_ > 2 * waitTime_ + 2000 & count_ < 2 * waitTime_ + 5000){
      // desired_contact_force.setZero();
      // for (int i = 0; i < desired_contact_force.size() / 3; i++){
      //     double z_value = desired_init_contact_force[i * 3 + 2];
      //     desired_contact_force[i * 3 + 2] = 150;
      // } 

      desired_contact_force_[0] = 0 + 30 * std::sin((count_ - 2 * waitTime_ - 2000)/200);
      // desired_contact_force_[0] = 130 * std::sin((count_ - 2 * waitTime_ - 2000)/2000);
      desired_contact_force_[1] = 0 + 20 * std::sin((count_ - 2 * waitTime_ - 2000)/1000);
      desired_contact_force_[2] = 140 - 50 * std::sin((count_ - 2 * waitTime_ - 2000)/500);
      // desired_contact_force_[2] = 140;
      // desired_contact_force_[3] = 0;
      // desired_contact_force_[4] = 0;
      // desired_contact_force_[5] = 150;
      // desired_contact_force_[6] = 0;
      // desired_contact_force_[7] = 0;
      // desired_contact_force_[8] = 150;
      // desired_contact_force_[9] = 0;
      // desired_contact_force_[10] = 0;
      // desired_contact_force_[11] = 150;

      // Eigen::VectorXd force_drift(desired_contact_force.rows());
      // force_drift.setZero();
      // force_drift[2] = 0;
      // desired_contact_force = desired_init_contact_force + force_drift;   
      // std::cout << "desired_contact_force: \n" << desired_contact_force << "\n";  
  }
  // else{
  //     desired_contact_force_[0] = 0;
  //     desired_contact_force_[1] = 0;
  //     desired_contact_force_[2] = 140; 
  // }

  // // Used for the step force
  // Eigen::VectorXd desired_contact_force_(3);
  // desired_contact_force_ << 0, 0, 100; // TODO: Hardcoded

  // // Eigen::VectorXd tau_tot_des = (Identity_matrix - Projection_matrix_)
  // //                                 * (Inertial_matrix_ * M_bar_.inverse() * (torque_command_ 
  // //                                 - Projection_matrix_ * h_vector_ + d_projection_matrix_ * anymal_dq_mrt_pinocchio_ + Projection_matrix_ * Jacobian_base_.transpose() * F_base_ext_est_) + h_vector_)
  // //                                 - (Identity_matrix - Projection_matrix_) * Jacobian_base_.transpose() * F_base_ext_est_;
  // // Eigen::VectorXd desired_contact_force_ = pseudoInverse(Jacobian_contact_.transpose()) * tau_tot_des;
  // if (count_ == 2 * waitTime_ + 2000){
  //     desired_init_contact_force = desired_contact_force_;
  // }
  // if (count_ > 2 * waitTime_ + 4000 & count_ <= 2 * waitTime_ + 6000){
  //     desired_contact_force_[0] = 0;
  //     desired_contact_force_[1] = 0;
  //     desired_contact_force_[2] = 130;
  // }
  // else if (count_ > 2 * waitTime_ + 6000 & count_ <= 2 * waitTime_ + 8000){
  //     desired_contact_force_[0] = 0;
  //     desired_contact_force_[1] = 0;
  //     desired_contact_force_[2] = 160; 
  // }
  // else if (count_ > 2 * waitTime_ + 8000 & count_ <= 2 * waitTime_ + 10000){
  //     desired_contact_force_[0] = 0;
  //     desired_contact_force_[1] = 0;
  //     desired_contact_force_[2] = 130; 
  // }
  // else{
  //     desired_contact_force_[0] = 0;
  //     desired_contact_force_[1] = 0;
  //     desired_contact_force_[2] = 100;                 
  // }

  // WBC.desired_LFFoot_forceMsg_.x = desired_contact_force_[0], WBC.desired_LFFoot_forceMsg_.y = desired_contact_force_[1], WBC.desired_LFFoot_forceMsg_.z = desired_contact_force_[2];
  // WBC.desired_RFFoot_forceMsg_.x = desired_contact_force_[3], WBC.desired_RFFoot_forceMsg_.y = desired_contact_force_[4], WBC.desired_RFFoot_forceMsg_.z = desired_contact_force_[5];
  // WBC.desired_LHFoot_forceMsg_.x = desired_contact_force_[6], WBC.desired_LHFoot_forceMsg_.y = desired_contact_force_[7], WBC.desired_LHFoot_forceMsg_.z = desired_contact_force_[8];
  // WBC.desired_RHFoot_forceMsg_.x = desired_contact_force_[9], WBC.desired_RHFoot_forceMsg_.y = desired_contact_force_[10], WBC.desired_RHFoot_forceMsg_.z = desired_contact_force_[11];            

  Eigen::VectorXd torque_c_ = (Identity_matrix - Projection_matrix_)
                              * (Inertial_matrix_ * M_bar_.inverse() * (torque_command_ 
                              - Projection_matrix_ * h_vector_ + d_projection_matrix_ * measured_v_ + Projection_matrix_ * Jacobian_base_.transpose() * F_base_ext_est_) + h_vector_)
                              - (Identity_matrix - Projection_matrix_) * Jacobian_base_.transpose() * F_base_ext_est_
                              - Jacobian_contact_force_control_.transpose() * desired_contact_force_;
                              // - Jacobian_contact_.transpose() * desired_contact_force_;
  Eigen::VectorXd tau_tot_d = torque_command_ + torque_c_;
  Eigen::VectorXd tau_const = tau_tot_d - Projection_matrix_ * Selection_matrix_support_ * QPSolution;
  // Eigen::VectorXd tau_const = torque_c_;

  Eigen::MatrixXd second_Hessian_matrix = 2 * Selection_matrix_force_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_force_;
  // Eigen::MatrixXd second_Hessian_matrix = 2 * Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_;
  // Eigen::MatrixXd second_Hessian_matrix = 2 * Selection_matrix_.transpose() * (Inertial_matrix_.inverse()).transpose() * Jacobian_contact_.transpose() * Jacobian_contact_ * Inertial_matrix_.inverse() * Selection_matrix_;
  // Eigen::MatrixXd second_Hessian_matrix = 2 * Selection_matrix_.transpose() * (Projection_matrix_.transpose() * (M_bar_.inverse()).transpose() * Jacobian_contact_.transpose() * Jacobian_contact_ * M_bar_.inverse() * Projection_matrix_) * Selection_matrix_;
  second_Hessian_matrix = (second_Hessian_matrix + second_Hessian_matrix.transpose())/2;
  Eigen::VectorXd second_f_vector = -2 * Selection_matrix_force_.transpose() * (Identity_matrix - Projection_matrix_).transpose() * tau_const;

  // Eigen::VectorXd second_lowerBound(info_.actuatedDofNum + 5 * contact_num_for_force_control_), second_upperBound(info_.actuatedDofNum + 5 * contact_num_for_force_control_);
  // Eigen::MatrixXd second_Constraint_matrix(info_.actuatedDofNum + 5 * contact_num_for_force_control_, info_.actuatedDofNum);  
  // Eigen::VectorXd second_lowerBound(2 * info_.actuatedDofNum + 5 * (contact_num_for_support_ + contact_num_for_force_control_)), second_upperBound(2 * info_.actuatedDofNum + 5 * (contact_num_for_support_ + contact_num_for_force_control_));
  // Eigen::MatrixXd second_Constraint_matrix(2 * info_.actuatedDofNum + 5 * (contact_num_for_support_ + contact_num_for_force_control_), info_.actuatedDofNum);    
  Eigen::VectorXd second_lowerBound(info_.actuatedDofNum + 5 * (contact_num_for_support_ + contact_num_for_force_control_)), second_upperBound(info_.actuatedDofNum + 5 * (contact_num_for_support_ + contact_num_for_force_control_));
  Eigen::MatrixXd second_Constraint_matrix(info_.actuatedDofNum + 5 * (contact_num_for_support_ + contact_num_for_force_control_), info_.actuatedDofNum); 

  // if (contact_num_for_force_control_ != 0){
  //     Eigen::MatrixXd W_f_matrix = -FrictionConstraint_for_force_control_matrix_ * pseudoInverse(Jacobian_contact_force_control_.transpose()) * (Identity_matrix - Projection_matrix_) * Selection_matrix_force_;
  //     Eigen::VectorXd rho_f_matrix_ = FrictionConstraint_for_force_control_matrix_ * pseudoInverse(Jacobian_contact_force_control_.transpose()) * (Identity_matrix - Projection_matrix_) 
  //                                     * ((Inertial_matrix_ * M_bar_.inverse() * (-Projection_matrix_ * h_vector_ + d_projection_matrix_ * anymal_dq_mrt_pinocchio_) + h_vector_) 
  //                                     + (Inertial_matrix_ * M_bar_.inverse() * Projection_matrix_ - Identity_matrix) 
  //                                     * (Jacobian_base_.transpose() * F_base_ext_est_));

  //     second_Constraint_matrix.topRows(12) = Eigen::MatrixXd::Identity(12, 12);    // Torque limit theoratical not correct
  //     // second_Constraint_matrix.topRows(info_.actuatedDofNum) = Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_;    // Torque limit theoratical correct
  //     // second_Constraint_matrix.topRows(info_.actuatedDofNum) = Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_force_;    // Torque limit theoratical correct
  //     // second_Constraint_matrix.middleRows(info_.actuatedDofNum, info_.actuatedDofNum) = Selection_matrix_support_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_support_;    // Torque limit theoratical correct
  //     // second_Constraint_matrix.middleRows(2 * info_.actuatedDofNum, 5 * contact_num_for_force_control_) = W_f_matrix;
  //     // second_Constraint_matrix.middleRows(info_.actuatedDofNum, 5 * contact_num_for_force_control_) = W_f_matrix;
  //     second_Constraint_matrix.bottomRows(5 * contact_num_for_force_control_) = W_f_matrix;
      
  //     // second_lowerBound.head(12) = TorqueLowerBound_ - QPSolution;    // Torque limit theoratical not correct
  //     second_lowerBound.head(info_.actuatedDofNum) = TorqueLowerBound_ - Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
  //     // second_lowerBound.head(info_.actuatedDofNum) = TorqueLowerBound_;   // Torque limit theoratical correct
  //     // second_lowerBound.segment(info_.actuatedDofNum, info_.actuatedDofNum) = TorqueLowerBound_ - Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
  //     // second_lowerBound.segment(2 * info_.actuatedDofNum, 5 * contact_num_for_force_control_) = low_D_f_vector_ - rho_f_matrix_;
  //     // second_lowerBound.segment(info_.actuatedDofNum, 5 * contact_num_for_force_control_) = low_D_f_vector_ - rho_f_matrix_;
  //     second_lowerBound.tail(5 * contact_num_for_force_control_) = low_D_f_vector_ - rho_f_matrix_;
      
  //     // second_upperBound.head(12) = TorqueUpperBound_ - QPSolution;   // Torque limit theoratical not correct
  //     second_upperBound.head(info_.actuatedDofNum) = TorqueUpperBound_ - Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
  //     // second_upperBound.head(info_.actuatedDofNum) = TorqueUpperBound_;   // Torque limit theoratical correct
  //     // second_upperBound.segment(info_.actuatedDofNum, info_.actuatedDofNum) = TorqueUpperBound_ - Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
  //     // second_upperBound.segment(2 * info_.actuatedDofNum, 5 * contact_num_for_force_control_) = high_D_f_vector_ - rho_f_matrix_;
  //     // second_upperBound.segment(info_.actuatedDofNum, 5 * contact_num_for_force_control_) = high_D_f_vector_ - rho_f_matrix_;
  //     second_upperBound.tail(5 * contact_num_for_force_control_) = high_D_f_vector_ - rho_f_matrix_;         
  // }
  // else{
  //     second_Constraint_matrix.topRows(12) = Eigen::MatrixXd::Identity(12, 12);    // Torque limit theoratical not correct
  //     // second_Constraint_matrix = Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_force_;    // Torque limit theoratical correct
  //     // second_Constraint_matrix.bottomRows(info_.actuatedDofNum) = Selection_matrix_support_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_support_;    // Torque limit theoratical correct
  //     // second_Constraint_matrix.bottomRows(info_.actuatedDofNum) = Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_;    // Torque limit theoratical correct

  //     // second_lowerBound.head(12) = TorqueLowerBound_ - QPSolution;    // Torque limit theoratical not correct
  //     // second_lowerBound.head(info_.actuatedDofNum) = TorqueLowerBound_;   // Torque limit theoratical correct
  //     // second_lowerBound.tail(info_.actuatedDofNum) = TorqueLowerBound_ - Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
  //     second_lowerBound = TorqueLowerBound_ - Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      
  //     // second_upperBound.head(12) = TorqueUpperBound_ - QPSolution;   // Torque limit theoratical not correct
  //     // second_upperBound.head(info_.actuatedDofNum) = TorqueUpperBound_;   // Torque limit theoratical correct
  //     // second_upperBound.tail(info_.actuatedDofNum) = TorqueUpperBound_ - Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct                  
  //     second_upperBound = TorqueUpperBound_ - Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct                  
  // }

  if (contact_num_for_force_control_ != 0 & contact_num_for_support_ != 0){
      Eigen::MatrixXd W_f_matrix = -FrictionConstraint_for_force_control_matrix_ * pseudoInverse(Jacobian_contact_force_control_.transpose()) * (Identity_matrix - Projection_matrix_) * Selection_matrix_force_;
      Eigen::VectorXd rho_f_matrix_ = FrictionConstraint_for_force_control_matrix_ * pseudoInverse(Jacobian_contact_force_control_.transpose()) * (Identity_matrix - Projection_matrix_) 
                                      * ((Inertial_matrix_ * M_bar_.inverse() * (-Projection_matrix_ * h_vector_ + d_projection_matrix_ * measured_v_) + h_vector_) 
                                      + (Inertial_matrix_ * M_bar_.inverse() * Projection_matrix_ - Identity_matrix) 
                                      * (Jacobian_base_.transpose() * F_base_ext_est_));

      Eigen::MatrixXd W_s_matrix = -FrictionConstraint_for_support_matrix_ * pseudoInverse(Jacobian_contact_support_.transpose()) * (Identity_matrix - Projection_matrix_) * Selection_matrix_support_;
      Eigen::VectorXd rho_s_matrix_ = FrictionConstraint_for_support_matrix_ * pseudoInverse(Jacobian_contact_support_.transpose()) * (Identity_matrix - Projection_matrix_) 
                                      * ((Inertial_matrix_ * M_bar_.inverse() * (Projection_matrix_ * Selection_matrix_support_ * QPSolution - Projection_matrix_ * h_vector_ + d_projection_matrix_ * measured_v_) + h_vector_) 
                                      + (Inertial_matrix_ * M_bar_.inverse() * Projection_matrix_ - Identity_matrix) 
                                      * (Jacobian_base_.transpose() * F_base_ext_est_));

      // second_Constraint_matrix.topRows(12) = Eigen::MatrixXd::Identity(12, 12);    // Torque limit theoratical not correct
      // second_Constraint_matrix.topRows(info_.actuatedDofNum) = Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_;    // Torque limit theoratical correct
      second_Constraint_matrix.topRows(info_.actuatedDofNum) = Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_force_;    // Torque limit theoratical correct
      // second_Constraint_matrix.middleRows(info_.actuatedDofNum, info_.actuatedDofNum) = Selection_matrix_support_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_support_;    // Torque limit theoratical correct
      // second_Constraint_matrix.middleRows(2 * info_.actuatedDofNum, 5 * contact_num_for_force_control_) = W_f_matrix;
      second_Constraint_matrix.middleRows(info_.actuatedDofNum, 5 * contact_num_for_force_control_) = W_f_matrix;
      second_Constraint_matrix.bottomRows(5 * contact_num_for_support_) = W_s_matrix;
      
      // second_lowerBound.head(12) = TorqueLowerBound_ - QPSolution;    // Torque limit theoratical not correct
      second_lowerBound.head(info_.actuatedDofNum) = TorqueLowerBound_ - Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      // second_lowerBound.head(info_.actuatedDofNum) = TorqueLowerBound_;   // Torque limit theoratical correct
      // second_lowerBound.segment(info_.actuatedDofNum, info_.actuatedDofNum) = TorqueLowerBound_ - Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      // second_lowerBound.segment(2 * info_.actuatedDofNum, 5 * contact_num_for_force_control_) = low_D_f_vector_ - rho_f_matrix_;
      second_lowerBound.segment(info_.actuatedDofNum, 5 * contact_num_for_force_control_) = low_D_f_vector_ - rho_f_matrix_;
      second_lowerBound.tail(5 * contact_num_for_support_) = low_D_m_vector_ - rho_s_matrix_;
      
      // second_upperBound.head(12) = TorqueUpperBound_ - QPSolution;   // Torque limit theoratical not correct
      second_upperBound.head(info_.actuatedDofNum) = TorqueUpperBound_ - Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      // second_upperBound.head(info_.actuatedDofNum) = TorqueUpperBound_;   // Torque limit theoratical correct
      // second_upperBound.segment(info_.actuatedDofNum, info_.actuatedDofNum) = TorqueUpperBound_ - Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      // second_upperBound.segment(2 * info_.actuatedDofNum, 5 * contact_num_for_force_control_) = high_D_f_vector_ - rho_f_matrix_;
      second_upperBound.segment(info_.actuatedDofNum, 5 * contact_num_for_force_control_) = high_D_f_vector_ - rho_f_matrix_;
      second_upperBound.tail(5 * contact_num_for_support_) = high_D_m_vector_ - rho_s_matrix_;         
  }
  else if (contact_num_for_force_control_ == 0 & contact_num_for_support_ != 0){
      Eigen::MatrixXd W_s_matrix = -FrictionConstraint_for_support_matrix_ * pseudoInverse(Jacobian_contact_support_.transpose()) * (Identity_matrix - Projection_matrix_) * Selection_matrix_support_;
      Eigen::VectorXd rho_s_matrix_ = FrictionConstraint_for_support_matrix_ * pseudoInverse(Jacobian_contact_support_.transpose()) * (Identity_matrix - Projection_matrix_) 
                                      * ((Inertial_matrix_ * M_bar_.inverse() * (Projection_matrix_ * Selection_matrix_support_ * QPSolution - Projection_matrix_ * h_vector_ + d_projection_matrix_ * measured_v_) + h_vector_) 
                                      + (Inertial_matrix_ * M_bar_.inverse() * Projection_matrix_ - Identity_matrix) 
                                      * (Jacobian_base_.transpose() * F_base_ext_est_));

      // second_Constraint_matrix.topRows(12) = Eigen::MatrixXd::Identity(12, 12);    // Torque limit theoratical not correct
      second_Constraint_matrix.topRows(info_.actuatedDofNum) = Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_force_;    // Torque limit theoratical correct
      // second_Constraint_matrix.topRows(info_.actuatedDofNum) = Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_;    // Torque limit theoratical correct
      // second_Constraint_matrix.middleRows(info_.actuatedDofNum, info_.actuatedDofNum) = Selection_matrix_support_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_support_;    // Torque limit theoratical correct
      second_Constraint_matrix.bottomRows(5 * contact_num_for_support_) = W_s_matrix;
      
      // second_lowerBound.head(12) = TorqueLowerBound_ - QPSolution;    // Torque limit theoratical not correct
      second_lowerBound.head(info_.actuatedDofNum) = TorqueLowerBound_ - Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      // second_lowerBound.head(info_.actuatedDofNum) = TorqueLowerBound_;   // Torque limit theoratical correct
      // second_lowerBound.segment(info_.actuatedDofNum, info_.actuatedDofNum) = TorqueLowerBound_ - Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      second_lowerBound.tail(5 * contact_num_for_support_) = low_D_m_vector_ - rho_s_matrix_;
      
      // second_upperBound.head(12) = TorqueUpperBound_ - QPSolution;   // Torque limit theoratical not correct
      second_upperBound.head(info_.actuatedDofNum) = TorqueUpperBound_ - Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      // second_upperBound.head(info_.actuatedDofNum) = TorqueUpperBound_;   // Torque limit theoratical correct
      // second_upperBound.segment(info_.actuatedDofNum, info_.actuatedDofNum) = TorqueUpperBound_ - Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      second_upperBound.tail(5 * contact_num_for_support_) = high_D_m_vector_ - rho_s_matrix_;                
  }
  else if (contact_num_for_force_control_ != 0 & contact_num_for_support_ == 0){
      Eigen::MatrixXd W_f_matrix = -FrictionConstraint_for_force_control_matrix_ * pseudoInverse(Jacobian_contact_force_control_.transpose()) * (Identity_matrix - Projection_matrix_) * Selection_matrix_force_;
      Eigen::VectorXd rho_f_matrix_ = FrictionConstraint_for_force_control_matrix_ * pseudoInverse(Jacobian_contact_force_control_.transpose()) * (Identity_matrix - Projection_matrix_) 
                                      * ((Inertial_matrix_ * M_bar_.inverse() * (-Projection_matrix_ * h_vector_ + d_projection_matrix_ * measured_v_) + h_vector_) 
                                      + (Inertial_matrix_ * M_bar_.inverse() * Projection_matrix_ - Identity_matrix) 
                                      * (Jacobian_base_.transpose() * F_base_ext_est_));

      // second_Constraint_matrix.topRows(12) = Eigen::MatrixXd::Identity(12, 12);    // Torque limit theoratical not correct
      // second_Constraint_matrix.topRows(info_.actuatedDofNum) = Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_;    // Torque limit theoratical correct
      second_Constraint_matrix.topRows(info_.actuatedDofNum) = Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_force_;    // Torque limit theoratical correct
      // second_Constraint_matrix.middleRows(info_.actuatedDofNum, info_.actuatedDofNum) = Selection_matrix_support_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_support_;    // Torque limit theoratical correct
      second_Constraint_matrix.bottomRows(5 * contact_num_for_force_control_) = W_f_matrix;
      
      // second_lowerBound.head(12) = TorqueLowerBound_ - QPSolution;    // Torque limit theoratical not correct
      second_lowerBound.head(info_.actuatedDofNum) = TorqueLowerBound_ - Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      // second_lowerBound.head(info_.actuatedDofNum) = TorqueLowerBound_;   // Torque limit theoratical correct
      // second_lowerBound.segment(info_.actuatedDofNum, info_.actuatedDofNum) = TorqueLowerBound_ - Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      second_lowerBound.tail(5 * contact_num_for_force_control_) = low_D_f_vector_ - rho_f_matrix_;
      
      // second_upperBound.head(12) = TorqueUpperBound_ - QPSolution;   // Torque limit theoratical not correct
      second_upperBound.head(info_.actuatedDofNum) = TorqueUpperBound_ - Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      // second_upperBound.head(info_.actuatedDofNum) = TorqueUpperBound_;   // Torque limit theoratical correct
      // second_upperBound.segment(info_.actuatedDofNum, info_.actuatedDofNum) = TorqueUpperBound_ - Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      second_upperBound.tail(5 * contact_num_for_force_control_) = high_D_f_vector_ - rho_f_matrix_;                 
  }
  else{
      // second_Constraint_matrix.topRows(12) = Eigen::MatrixXd::Identity(12, 12);    // Torque limit theoratical not correct
      second_Constraint_matrix.topRows(info_.actuatedDofNum) = Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_force_;    // Torque limit theoratical correct
      // second_Constraint_matrix.bottomRows(info_.actuatedDofNum) = Selection_matrix_support_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_support_;    // Torque limit theoratical correct
      // second_Constraint_matrix.bottomRows(info_.actuatedDofNum) = Selection_matrix_.transpose() * (Identity_matrix - Projection_matrix_) * Selection_matrix_;    // Torque limit theoratical correct

      // second_lowerBound.head(12) = TorqueLowerBound_ - QPSolution;    // Torque limit theoratical not correct
      // second_lowerBound.head(info_.actuatedDofNum) = TorqueLowerBound_;   // Torque limit theoratical correct
      // second_lowerBound.tail(info_.actuatedDofNum) = TorqueLowerBound_ - Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      second_lowerBound.tail(info_.actuatedDofNum) = TorqueLowerBound_ - Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct
      
      // second_upperBound.head(12) = TorqueUpperBound_ - QPSolution;   // Torque limit theoratical not correct
      // second_upperBound.head(info_.actuatedDofNum) = TorqueUpperBound_;   // Torque limit theoratical correct
      // second_upperBound.tail(info_.actuatedDofNum) = TorqueUpperBound_ - Selection_matrix_support_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct                  
      second_upperBound.tail(info_.actuatedDofNum) = TorqueUpperBound_ - Selection_matrix_.transpose() * Projection_matrix_ * Selection_matrix_support_ * QPSolution;   // Torque limit theoratical correct                  
  }

  unsigned int second_numOfVar = second_Hessian_matrix.rows();
  unsigned int second_numOfRows = second_Constraint_matrix.rows();

  Eigen::SparseMatrix<double> second_hessian(info_.actuatedDofNum, info_.actuatedDofNum);
  for(int i=0; i<second_numOfVar; i++){          
      for(int j=0; j<second_numOfVar; j++){     
          second_hessian.insert(i, j) = second_Hessian_matrix(i, j);
      }
  }

  Eigen::SparseMatrix<double> second_linearMatrix(second_numOfRows, info_.actuatedDofNum);
  for(int i=0; i<second_numOfRows; i++){         
      for(int j=0; j<second_numOfVar; j++){      
          second_linearMatrix.insert(i, j) = second_Constraint_matrix(i, j); 
      }
  }

  // Eigen::VectorXd shifted_lowerBound = WBC.lowerBound_ - WBC.Constraint_matrix_ * QPSolution;
  // Eigen::VectorXd shifted_upperBound = WBC.upperBound_ - WBC.Constraint_matrix_ * QPSolution;

  Eigen::VectorXd second_gradient(info_.actuatedDofNum);
  second_gradient = second_f_vector;

  // OsqpEigen::Solver second_solver;
  // solver.settings()->setVerbosity(false);
  // solver.settings()->setAlpha(1.0);

  // REQUIRE_FALSE(solver.data()->setHessianMatrix(second_hessian));
  solver.clearSolver();
  solver.data()->clearHessianMatrix();
  solver.data()->clearLinearConstraintsMatrix();
  solver.settings()->setVerbosity(false);
  solver.settings()->setAlpha(1.0);
  solver.data()->setNumberOfVariables(info_.actuatedDofNum);
  solver.data()->setNumberOfConstraints(second_numOfRows);

  if(!solver.data()->setHessianMatrix(second_hessian)) return vector_t();
  if(!solver.data()->setGradient(second_gradient)) return vector_t();
  if(!solver.data()->setLinearConstraintsMatrix(second_linearMatrix)) return vector_t();
  if(!solver.data()->setLowerBound(second_lowerBound)) return vector_t(); 
  if(!solver.data()->setUpperBound(second_upperBound)) return vector_t(); 

  if(!solver.initSolver()) return vector_t();

  solver.solve();

  Eigen::VectorXd second_QPSolution = solver.getSolution();      

  Eigen::VectorXd total_QPSolution = Selection_matrix_.transpose() * (Projection_matrix_ * Selection_matrix_support_ * QPSolution + (Identity_matrix - Projection_matrix_) * Selection_matrix_force_ * second_QPSolution);
  return total_QPSolution;
}

}  // namespace legged