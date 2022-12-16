#!/usr/bin/env python
import copy
import numpy as np

import pydrake.all as pd

dt = 5e-3
tc = 1e-1
N = 10
T = (N + 1) * tc

NR = 5

NQ = 2
NV = 2
NX = NQ + NV
NU = 1

qd = np.array([np.pi, 0.0])
wui = 1e-3
wvi = 1e-4
wqi = 1e-1
wqf = 1e0

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


def rollout(diagram, simulator, context, x0, U0, perturb=True):
    context.SetTime(0.0)
    simulator.Initialize()
    context.SetDiscreteState(x0)
    sim_t = 0.0
    X = np.tile(x0, (N+1, 1))
    U = U0
    for i in range(N):
        if perturb:
            U[:, i] = np.random.normal(U0[:, i], 0.1, (NU, 1))
        diagram.get_input_port(0).FixValue(context, U[:, i])
        sim_t = sim_t + tc
        simulator.AdvanceTo(sim_t)
        X[i + 1] = state.CopyToVector()
    return X, U


# Create a diagram builder, a plant, and a scene.
builder = pd.DiagramBuilder()
plant, scene = pd.AddMultibodyPlantSceneGraph(builder, dt)
# Load the plant.
file_name = pd.FindResourceOrThrow(
    "drake/examples/acrobot/Acrobot_no_collision.urdf")
pd.Parser(plant).AddModels(file_name)
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link"));
plant.Finalize()
# Add a meshcat visualizer.
meshcat = pd.StartMeshcat()
visualizer = pd.MeshcatVisualizer.AddToBuilder(
    builder=builder, scene_graph=scene, meshcat=meshcat)
# Set the input port to the plant's actuation.
builder.ExportInput(plant.get_actuation_input_port())
# Build the diagram and load into a simulator.
diagram = builder.Build()
simulator = pd.Simulator(diagram)

q0 = np.array([np.pi, 0.0])
v0 = np.zeros(2)
x0 = np.hstack((q0, v0))
context = simulator.get_mutable_context()
state = context.get_mutable_discrete_state_vector()

U0 = np.zeros((NU, N))

min_cost = 1e20
Uopt = None
Xopt = None
for i in range(NR):
    X, U = rollout(diagram, simulator, context, x0, U0)
    cost = eval_cost(X, U)
    if cost <= min_cost:
        min_cost = copy.copy(cost)
        Uopt = copy.copy(U)
        Xopt = copy.copy(X)
        U0 = copy.copy(U)
    print(f"X:\n{X}")
    print(f"U:\n{U}")
    print(f"Cost: {cost}")

print("Best sample:")
print(f"X:\n{Xopt}")
print(f"U:\n{Uopt}")
print(f"Cost: {min_cost}")


visualizer.StartRecording(set_transforms_while_recording=False)
X, U = rollout(diagram, simulator, context, x0, Uopt, False)
visualizer.PublishRecording()

input()
