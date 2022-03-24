// @file
// Benchmarks for PositionConstraint.
//

#include <benchmark/benchmark.h>

#include "drake/common/find_resource.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/constraint.h"
#include "drake/solvers/solve.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"

namespace drake {
namespace systems {
namespace trajectory_optimization {
namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using solvers::BoundingBoxConstraint;
using solvers::VectorXDecisionVariable;
using symbolic::Expression;
using trajectories::PiecewisePolynomial;

// NOLINTNEXTLINE(runtime/references) cpplint disapproves of gbench choices.
static void IiwaDirectCollocation(benchmark::State& state) {
  multibody::MultibodyPlant<double> plant(0.0);
  geometry::SceneGraph<double> scene_graph;
  plant.RegisterAsSourceForSceneGraph(&scene_graph);
  const std::string file_name = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/iiwa7/"
      "iiwa7_no_collision.sdf");
  multibody::Parser(&plant).AddModelFromFile(file_name);
  plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"));
  plant.Finalize();

  auto context = plant.CreateDefaultContext();
  const int kNumTimeSteps{21};
  const double kMinTimeStep{0.1};
  const double kMaxTimeStep{0.4};

  for (auto _ : state) {
    // Only this code gets timed
    DirectCollocation dircol(&plant, *context, kNumTimeSteps, kMinTimeStep,
                             kMaxTimeStep,
                             plant.get_actuation_input_port().get_index());
    solvers::MathematicalProgram& prog = dircol.prog();

    const VectorXDecisionVariable& u = dircol.input();
    const VectorXDecisionVariable& x = dircol.state();

    dircol.AddEqualTimeIntervalsConstraints();

    // Initial conditions.
    VectorXd q0(7);
    q0 << 0.0, 0.1, 0, -1.2, 0, 1.6, 0;
    VectorXd x0 = VectorXd::Zero(14);
    x0.head<7>() << q0;
    prog.AddBoundingBoxConstraint(x0, x0, dircol.initial_state());

    // Final conditions.
    VectorXd qf(7);
    qf << 0.0, -0.1, 0, 1.2, 0, -1.6, 0;
    VectorXd xf = VectorXd::Zero(14);
    xf.head<7>() << qf;
    prog.AddBoundingBoxConstraint(xf, xf, dircol.final_state());

    // Start and end with zero torque.
    prog.AddBoundingBoxConstraint(0, 0, dircol.input(0));
    prog.AddBoundingBoxConstraint(0, 0, dircol.input(kNumTimeSteps - 1));

    // Position, velocity, and effort constraints.
    dircol.AddConstraintToAllKnotPoints(
        std::make_shared<BoundingBoxConstraint>(plant.GetPositionLowerLimits(),
                                                plant.GetPositionUpperLimits()),
        dircol.state().head<7>());
    dircol.AddConstraintToAllKnotPoints(
        std::make_shared<BoundingBoxConstraint>(plant.GetVelocityLowerLimits(),
                                                plant.GetVelocityUpperLimits()),
        dircol.state().tail<7>());
    dircol.AddConstraintToAllKnotPoints(
        std::make_shared<BoundingBoxConstraint>(plant.GetEffortLowerLimits(),
                                                plant.GetEffortUpperLimits()),
        dircol.input());

    // Cost on input "effort".
    dircol.AddRunningCost(10 * u.cast<Expression>().squaredNorm());
    // Cost on velocity.
    dircol.AddRunningCost(x.tail<7>().cast<Expression>().squaredNorm());
    // Cost on the total duration.
    dircol.AddFinalCost(dircol.time().cast<Expression>());

    MatrixXd X_guess(14, 2);
    X_guess << x0, xf;
    PiecewisePolynomial<double> x_trajectory_guess =
        PiecewisePolynomial<double>::FirstOrderHold(Eigen::Vector2d{0., 4.},
                                                    X_guess);
    dircol.SetInitialTrajectory(PiecewisePolynomial<double>(),
                                x_trajectory_guess);

    auto result = solvers::Solve(prog);
    DRAKE_DEMAND(result.is_success());
  }
}

}  // namespace
}  // namespace trajectory_optimization
}  // namespace systems
}  // namespace drake

BENCHMARK(drake::systems::trajectory_optimization::IiwaDirectCollocation);
BENCHMARK_MAIN();
