#!/usr/bin/env python
import copy
import numpy as np

import pydrake.all as pd

# Deviation of the normal distribution.
SIGMA = 0.1
# Number of rollouts.
NR = 10

# Dynamic time step size [s].
dt = 1e-2
# Control time step size [s].
tc = 1e-1
# Prection horizon for the controller.
N = 20
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
Q0 = np.array([0., 0.])
V0 = np.array([0.0, 0.0])
X0 = np.hstack((Q0, V0))

# Cost weights.
wui = 1e-3
wvi = 1e-2
wvf = 1e0
wqi = 1e1
wqf = 1e2


def wrap_angles(q):
    return (q + np.pi) % (2*np.pi) - np.pi


def eval_cost(X, U):
    Q = X[:, :NQ]
    V = X[:, NV:]
    cost = 0.0
    for i in range(N):
        cost += wui * U[:, i].T @ U[:, i]
        cost += wvi * V[i].T @ V[i]
        cost += wqi * wrap_angles(qd - Q[i]).T @ wrap_angles(qd - Q[i])
    cost += wvf * V[N].T @ V[N]
    cost += wqf * wrap_angles(qd - Q[N]).T @ wrap_angles(qd - Q[N])
    return cost


def rollout(diagram, simulator, context, x0, U0, perturb=True):
    context.SetTime(0.0)
    simulator.Initialize()
    context.SetDiscreteState(x0)
    state = context.get_mutable_discrete_state_vector()
    sim_t = 0.0
    X = np.tile(x0, (N+1, 1))
    U = copy.copy(U0)
    for i in range(N):
        if perturb:
            U[:, i] = np.random.normal(U0[:, i], SIGMA, (NU, 1))
        diagram.get_input_port(0).FixValue(context, U[:, i])
        sim_t = sim_t + tc
        simulator.AdvanceTo(sim_t)
        X[i + 1] = state.CopyToVector()
        X[i + 1][:NQ] = X[i + 1][:NQ]
    return X, U

def run_controller(diagram, simulator, context, x0, U0):
    # Initialize.
    X, U = rollout(diagram, simulator, context, x0, U0, perturb=False)
    min_cost = eval_cost(X, U)
    Uopt = copy.copy(U0)
    # Also try the shifted control trajectory.
    U = np.delete(U0, 0, 1)
    U = np.hstack((U, U[:, [-1]]))
    X, _ = rollout(diagram, simulator, context, x0, U, perturb=False)
    cost = eval_cost(X, U)
    if cost < min_cost:
        min_cost = copy.copy(cost)
        U0 = copy.copy(U)
        Uopt = copy.copy(U)
    # Run the rollouts.
    for _ in range(NR):
        X, U = rollout(diagram, simulator, context, x0, U0)
        cost = eval_cost(X, U)
        if cost <= min_cost:
            min_cost = copy.copy(cost)
            Uopt = copy.copy(U)
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
    print(f"\tCurrent state: {x_curr}")
    # Run predictive sampling.
    Ups = run_controller(
        controller_diagram, controller_simulator, controller_context,
        x_curr, U0)
    # Apply the first control input.
    diagram.get_input_port(0).FixValue(context, Ups[:, 0])
    # Update the initial control trajectory.
    U0 = copy.copy(Ups)
    # Take a control time step.
    sim_t = sim_t + dt
    simulator.AdvanceTo(sim_t)
    print(f"Time: {sim_t} s")
    # Count the number of time steps.
    t_steps = t_steps + 1
    # input()

# Publish the recording and hang.
visualizer.PublishRecording()
input()
