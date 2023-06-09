import pdb

import copy
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


# Create a builder, a multibody plant, and a scene graph.
builder = DiagramBuilder()
dt = 1e-1
plant, sg = AddMultibodyPlantSceneGraph(builder, dt)

# Add the acrobot model.
file_name = FindResourceOrThrow(
    "drake/multibody/benchmarks/acrobot/acrobot.sdf")
Parser(plant).AddModels(file_name)
plant.Finalize()

# Get plant dimensions.
nq = plant.num_positions()
nv = plant.num_velocities()
nx = nq + nv
nu = plant.num_actuators()

# Set up visualization.
meshcat = StartMeshcat()
visualizer = MeshcatVisualizer.AddToBuilder(builder, sg, meshcat)

# Build the diagram.
diagram = builder.Build()
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(context)

# Set the initial state.
q0 = np.array([1., 0.])
v0 = np.array([0., 0.])
x0 = np.concatenate((q0, v0))
context.SetDiscreteState(x0)


# Create linearization system.
lin = LinSys(plant)

# Rollout.
N = 10

A = [np.zeros((2*nv, 2*nv)) for _ in range(N)]
B = [np.zeros((2*nv, nu)) for _ in range(N)]
U = np.zeros((N, nu))
X = np.tile(x0, (N+1, 1))
Xp = np.tile(x0, (N+1, 1))

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
    print(f"t: {context.get_time()} s, u: {u}")
    print(f"x act: {X[i+1]},")
    print(f"x est: {Xp[i+1]}")
    print(f"accuracy: {np.linalg.norm(X[i+1] - Xp[i+1])}")
    input("Continue?\n")


prog = MathematicalProgram()
dX = prog.NewContinuousVariables(rows=N+1, cols=nx, name='dx')
dU = prog.NewContinuousVariables(rows=N, cols=nu, name='du')

# Set the desired state.
xd = np.zeros(nx)

# Set the cost weights.
R = 1e-3*np.eye(nu)
Q = 1e0*np.eye(nx)
Qf = 1e2*np.eye(nx)

# Add running costs and integration constraints.
for i in range(N):
    prog.AddQuadraticErrorCost(Q=R, x_desired=-U[i], vars=dU[i])
    prog.AddQuadraticErrorCost(Q=Q, x_desired=xd-X[i], vars=dX[i])
    lin_dyn_eq_b = X[i+1] - A[i] @ X[i] - B[i] @ U[i]
    lin_dyn_eq_A = np.block([A[i], B[i], -np.eye(nx)])
    prog.AddLinearEqualityConstraint(Aeq=lin_dyn_eq_A, beq=lin_dyn_eq_b,
                                     vars=np.concatenate((dX[i], dU[i], dX[i+1])))
# Add terminal cost
prog.AddQuadraticErrorCost(Q=Qf, x_desired=xd-X[N], vars=dX[N])

solver = GurobiSolver()
result = solver.Solve(prog)

pdb.set_trace()
