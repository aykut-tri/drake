/// @file
///
/// Demo of LQR control of iiwa.

#include <iostream>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/primitives/affine_system.h"
#include "drake/visualization/visualization_config_functions.h"

namespace drake {

using multibody::AddMultibodyPlantSceneGraph;
using multibody::MultibodyPlant;
using multibody::Parser;
using systems::AffineSystem;
using systems::Context;
using systems::InputPort;
using systems::controllers::LinearQuadraticRegulator;

namespace examples {
namespace kuka_iiwa_arm {
namespace {

int do_main() {
  // Initialize the builder, the plant, and scene.
  systems::DiagramBuilder<double> builder;
  const double dt = 1e-3;
  auto [plant, scene] = AddMultibodyPlantSceneGraph(&builder, dt);

  // Create the plant.
  const std::string rel_sdf_path =
      "drake/manipulation/models/iiwa_description/sdf/"
      "iiwa14_no_collision.sdf";
  const std::string abs_sdf_path = FindResourceOrThrow(rel_sdf_path);
  Parser parser(&plant);
  parser.AddModels(abs_sdf_path);
  plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"));
  plant.Finalize();

  // Get the model properties.
  const int nq = plant.num_positions();
  const int nv = plant.num_velocities();
  const int nx = nq + nv;
  const int nu = plant.num_actuators();

  // Create an LQR controller.
  MultibodyPlant<double> lqr_plant(0);
  Parser lqr_parser(&lqr_plant);
  lqr_parser.AddModels(abs_sdf_path);
  lqr_plant.WeldFrames(lqr_plant.world_frame(),
                       lqr_plant.GetFrameByName("iiwa_link_0"));
  lqr_plant.Finalize();
  std::unique_ptr<Context<double>> lqr_context =
      lqr_plant.CreateDefaultContext();
  Eigen::VectorXd q0(nq);
  q0 << 0, 1.2, 0, -2, 0, -1.5, 0;
  lqr_plant.SetPositions(lqr_context.get(), q0);
  lqr_plant.SetVelocities(lqr_context.get(), Eigen::VectorXd::Zero(nv));
  const InputPort<double>& actuation_port =
      lqr_plant.get_actuation_input_port();
  auto g = lqr_plant.CalcGravityGeneralizedForces(*lqr_context);
  actuation_port.FixValue(lqr_context.get(), -g);

  // Setup the LQR weights.
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx);
  Q.topLeftCorner(nq, nq) *= 1e3;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu);

  // Create the controller.
  std::unique_ptr<AffineSystem<double>> lqr = LinearQuadraticRegulator(
      lqr_plant, *lqr_context, Q, R,
      Eigen::Matrix<double, 0, 0>::Zero() /* No cross state/control costs */,
      actuation_port.get_index());

  // Plug the controller into the diagram.
  auto controller = builder.AddSystem(std::move(lqr));
  builder.Connect(plant.get_state_output_port(), controller->get_input_port());
  builder.Connect(controller->get_output_port(),
                  plant.get_actuation_input_port());

  // Add a visualizer and build.
  visualization::AddDefaultVisualization(&builder);
  auto diagram = builder.Build();

  // Simulate.
  systems::Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(1.0);
  for (int i = 0; i < 5; i++) {
    simulator.get_mutable_context().SetTime(0.0);
    Eigen::VectorXd perturbed_x0(nx);
    perturbed_x0.head(nq) = q0 + Eigen::VectorXd::Random(nq);
    perturbed_x0.tail(nv) = Eigen::VectorXd::Random(nv);
    simulator.get_mutable_context()
        .get_mutable_discrete_state()
        .get_mutable_vector()
        .SetFromVector(perturbed_x0);

    simulator.Initialize();
    simulator.AdvanceTo(3.0);

    std::cout << "\nPress Enter to continue...\n";
    std::cin.get();
  }

  return 0;
}

}  // namespace
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::kuka_iiwa_arm::do_main();
}