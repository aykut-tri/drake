import pdb
import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    FindResourceOrThrow,
    GurobiSolver,
    Linearize,
    MathematicalProgram,
    MeshcatVisualizer,
    OsqpSolver,
    OutputPortSelection,
    Parser,
    Simulator,
    StartMeshcat,
)


class LinSys:
    def __init__(self, plant):
        # Get the plant.
        self.plant = plant
        # Create system-specific context.
        self.context = plant.CreateDefaultContext()
        # Get the actuation input port.
        self.input_port = plant.get_actuation_input_port()

    def linearize(self, x, u, eq_check_tol=1e1):
        # Set the state.
        self.context.SetDiscreteState(x)
        # Set the control input.
        self.input_port.FixValue(self.context, u)
        # Linearize the system about the current operating point.
        linsys = Linearize(system=self.plant, context=self.context,
                           input_port_index=self.input_port.get_index(),
                           output_port_index=OutputPortSelection.kNoOutput,
                           equilibrium_check_tolerance=eq_check_tol)
        return linsys.A(), linsys.B()


def eval_cost(Xd, X, U, N, R, Q, Qf):
    cost = 0.0
    for i in range(N):
        cost += (Xd[i] - X[i]).T @ Q @ (Xd[i] - X[i])
        cost += U[i].T @ R @ U[i]
    cost += (Xd[N] - X[N]).T @ Qf @ (Xd[N] - X[N])
    return cost

def rollout(simulator, plant, plant_context, x0, U, N, dt):
    # Reset the simulator.
    context = simulator.get_mutable_context()
    context.SetTime(0)
    simulator.Initialize()
    # Set the initial state.
    plant_context.SetDiscreteState(x0)
    # Set the trajectory to the initial state.
    X_out = np.tile(x0, (N+1, 1))
    # Rollout the control trajectory.
    for i in range(N):
        # Set the control input.
        plant.get_actuation_input_port().FixValue(plant_context, U[i])
        # Advance to the next time step.
        simulator.AdvanceTo((i+1)*dt)
        X_out[i+1] = plant.GetPositionsAndVelocities(plant_context)
    return X_out


if __name__ == "__main__":
    # Create a builder, a multibody plant, and a scene graph.
    builder = DiagramBuilder()
    dt = 1e-2
    plant, sg = AddMultibodyPlantSceneGraph(builder, dt)

    # Add the acrobot model.
    file_name = FindResourceOrThrow(
        "drake/multibody/benchmarks/acrobot/acrobot.sdf")
    Parser(plant).AddModels(file_name)
    plant.Finalize()

    # Get plant dimensions.
    NQ = plant.num_positions()
    NV = plant.num_velocities()
    NX = NQ + NV
    NU = plant.num_actuators()

    # Set up visualization.
    meshcat = StartMeshcat()
    visualizer = MeshcatVisualizer.AddToBuilder(builder, sg, meshcat)

    # Build the diagram.
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Set the initial state.
    q0 = np.array([0.0, 0.0])
    v0 = np.array([0.0, 0.0])
    x0 = np.concatenate((q0, v0))
    plant_context.SetDiscreteState(x0)

    # Create linearization system.
    lin = LinSys(plant)

    # Set the horizon length.
    N = 10
    # Initialize the nominal state and control trajectories.
    U = np.zeros((N, NU))
    X = np.tile(x0, (N+1, 1))
    Xp = np.tile(x0, (N+1, 1))

    # Initialize the derivative matrices.
    A = [np.zeros((2*NV, 2*NV)) for _ in range(N)]
    B = [np.zeros((2*NV, NU)) for _ in range(N)]

    # Check the linearization accuracy.
    simulator.Initialize()
    for i in range(N):
        # Select the state and control vectors for the time step.
        x = X[i]
        u = U[i]
        # Set the control input.
        plant.get_actuation_input_port().FixValue(plant_context, u)
        # Linearize about the current operating point.
        A[i], B[i] = lin.linearize(x=x, u=u)
        # Predict the next state using the linear approximation.
        Xp[i+1] = A[i] @ x + B[i] @ u
        # Print out.
        print(f"A:\n{A[i]}\nB:\n{B[i]}\n")
        # Advance to the next time step.
        simulator.AdvanceTo((i+1)*dt)
        # Get the actual next state.
        X[i+1] = plant.GetPositionsAndVelocities(plant_context)
        print(f"t: {context.get_time()} s, x: {x}, u: {u}")
        print(f"xnext est: {Xp[i+1]}")
        print(f"xnext act: {X[i+1]},")
        print(f"error: {np.linalg.norm(X[i+1] - Xp[i+1])}")

    # Create the mathematical program and change of state/control variables.
    prog = MathematicalProgram()
    dX = prog.NewContinuousVariables(rows=N+1, cols=NX, name='dx')
    dU = prog.NewContinuousVariables(rows=N, cols=NU, name='du')

    # Set the desired trajectory.
    xd = np.array([np.pi, 0.0, 0.0, 0.0])
    Xd = np.tile(xd, (N+1, 1))
    print(f"Desired state trajectory:\n{Xd}")

    # Set the cost weights.
    R = 1e-3*np.eye(NU)
    Q = 1e-1*np.eye(NX)
    Qf = 1e2*np.eye(NX)

    # Successive convexification.
    converged = False
    while not converged:
        # Create the subproblem by linearizing the dynamics.
        for i in range(N):
            # Linearize the dynamics about the previous solution.

            # Add running cost terms.
            prog.AddQuadraticErrorCost(Q=R, x_desired=-U[i], vars=dU[i])
            prog.AddQuadraticErrorCost(Q=Q, x_desired=Xd[i]-X[i], vars=dX[i])
            lin_dyn_eq_b = X[i+1] - A[i] @ X[i] - B[i] @ U[i]
            lin_dyn_eq_A = np.block([A[i], B[i], -np.eye(NX)])
            prog.AddLinearEqualityConstraint(Aeq=lin_dyn_eq_A, beq=lin_dyn_eq_b,
                                            vars=np.concatenate((dX[i], dU[i], dX[i+1])))
        # Add the terminal term.
        prog.AddQuadraticErrorCost(Q=Qf, x_desired=Xd[N]-X[N], vars=dX[N])

        # Solve the subproblem.
        solver = GurobiSolver()
        result = solver.Solve(prog)

        # Get the optimal change of variables.
        dX_sol = result.GetSolution(dX)
        dU_sol = (result.GetSolution(dU)).reshape(N, NU)

        # Evaluate the cost.
        L_sol = result.get_optimal_cost()
        X_sol = X + dX_sol
        U_sol = U + dU_sol
        L_eval = eval_cost(Xd, X_sol, U_sol, N, R, Q, Qf)

        # Print out the solution and the cost values for the subproblem.
        print("\nSolution:")
        print(f"dX:\n{dX_sol}\ndU:\n{dU_sol}\n")
        print(f"MP L: {L_sol}")
        print(f"My L: {L_eval}")

        # Rollout the updated control trajectory.
        X_rollout = rollout(simulator, plant, plant_context, x0, U_sol, N, dt)
        print(f"Rollout:\n{X_rollout}\n")

        # Quit directly.
        converged = True

    pdb.set_trace()
