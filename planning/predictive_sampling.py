#!/usr/bin/env python
import copy
import numpy as np

import pydrake.all as pd

# Deviation of the normal distribution.
SIGMA = 0.01
# Number of rollouts.
NR = 100

# Dynamic time step size [s].
dt = 5e-3
# Control time step size [s].
tc = 1e-2
# Prection horizon for the controller.
N = 10
# Total duration of a rollout [s].
T = (N + 1) * tc

# Plant properties.
NQ = 2
NV = 2
NX = NQ + NV
NU = 1

# Desired state and weights.
qd = np.array([np.pi, 0.0])

# Initial state.
Q0 = np.array([np.pi, 0.0])
V0 = np.array([0.0, 0.0])
X0 = np.hstack((Q0, V0))

# Cost weights.
wui = 1e-3
wvi = 1e0
wqi = 1e2
wqf = 1e6


def wrap_angles_rad(q):
    return (q + np.pi) % (2*np.pi) - np.pi


def eval_cost(X, U):
    Q = X[:, :NQ]
    V = X[:, NV:]
    cost = 0.0
    for i in range(N):
        cost += wui * U[:, i].T @ U[:, i]
        cost += wvi * V[i].T @ V[i]
        cost += wqi * (qd - Q[i]).T @ (qd - Q[i])
    cost += wvi * V[N].T @ V[N]
    cost += wqf * (qd - Q[N]).T @ (qd - Q[N])
    return cost


def rollout(diagram, simulator, context, x0, U0):
    context.SetTime(0.0)
    simulator.Initialize()
    context.SetDiscreteState(x0)
    state = context.get_mutable_discrete_state_vector()
    sim_t = 0.0
    X = np.tile(x0, (N+1, 1))
    U = copy.copy(U0)
    for i in range(N):
        U[:, i] = np.random.normal(U0[:, i], SIGMA, (NU, 1))
        diagram.get_input_port(0).FixValue(context, U[:, i])
        sim_t = sim_t + tc
        simulator.AdvanceTo(sim_t)
        X[i + 1] = state.CopyToVector()
        X[i + 1][:NQ] = wrap_angles_rad(X[i + 1][:NQ])
    return X, U

def run_controller(diagram, simulator, context, x0, U0):
    # Initialize.
    min_cost = 1e20
    Uopt = None
    Xopt = None
    # Run the rollouts.
    for i in range(NR):
        X, U = rollout(diagram, simulator, context, x0, U0)
        cost = eval_cost(X, U)
        if cost <= min_cost:
            min_cost = copy.copy(cost)
            Uopt = copy.copy(U)
            Xopt = copy.copy(X)
        # print(f"Rollout {i}\n")
        # print(f"Cost: {cost}")
    # Print the best sample.
    # print("Best sample:")
    # print(f"X:\n{Xopt}")
    # print(f"U:\n{Uopt}")
    print(f"\tMin. cost: {min_cost}")
    return Uopt

# Find the model file.
file_name = pd.FindResourceOrThrow(
    "drake/examples/acrobot/Acrobot_no_collision.urdf")

# Build a system for simulation.
builder = pd.DiagramBuilder()
plant, scene = pd.AddMultibodyPlantSceneGraph(builder, dt)
# Load the plant.
pd.Parser(plant).AddModels(file_name)
plant.WeldFrames(
    plant.world_frame(), plant.GetFrameByName("base_link"))
plant.Finalize()
# Set the input port to the plant's actuation.
builder.ExportInput(plant.get_actuation_input_port())
# Add a meshcat visualizer.
meshcat = pd.StartMeshcat()
visualizer = pd.MeshcatVisualizer.AddToBuilder(
    builder=builder, scene_graph=scene, meshcat=meshcat)
# Build the diagram and load into a simulator.
diagram = builder.Build()
simulator = pd.Simulator(diagram)

# Build a system for control.
builder = pd.DiagramBuilder()
controller_plant = builder.AddSystem(pd.MultibodyPlant(dt))
pd.Parser(controller_plant).AddModels(file_name)
controller_plant.WeldFrames(
    controller_plant.world_frame(),
    controller_plant.GetFrameByName("base_link"))
controller_plant.Finalize()
# Set the input port to the plant's actuation.
builder.ExportInput(controller_plant.get_actuation_input_port())
# Build the diagram and load into a simulator.
controller_diagram = builder.Build()
controller_simulator = pd.Simulator(controller_diagram)

# Get handles on the contexts and the simulation state.
context = simulator.get_mutable_context()
state = context.get_mutable_discrete_state_vector()
controller_context = controller_simulator.get_mutable_context()

# Set the initial state.
context.SetDiscreteState(X0)

# Set the initial control trajectory.
U0 = np.zeros((NU, N))

# Run and record a simulation.
visualizer.StartRecording(set_transforms_while_recording=True)
sim_t = 0.0
t_steps = 0
while t_steps < 1e4:
    # Get the current state.
    x_curr = state.CopyToVector()
    x_curr[:NQ] = wrap_angles_rad(x_curr[:NQ])
    print(f"\tCurrent state: {x_curr}")
    # Run predictive sampling.
    Ups = run_controller(
        controller_diagram, controller_simulator, controller_context,
        x_curr, U0)
    # Apply the first control input.
    diagram.get_input_port(0).FixValue(context, Ups[:, 0])
    # Take a control time step.
    sim_t = sim_t + dt
    simulator.AdvanceTo(sim_t)
    print(f"Time: {sim_t} s")
    # Shift the control trajectory.
    U0 = np.delete(Ups, 0, 1)
    U0 = np.hstack((U0, U0[:, [-1]]))
    # Count the number of time steps.
    t_steps = t_steps + 1
    # input()

# Publish the recording and hang.
visualizer.PublishRecording()
input()
