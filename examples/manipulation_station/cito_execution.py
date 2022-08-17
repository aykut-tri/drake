"""
This is an example for following the horizontal position of a box detected
by an optitrack setup with a Kuka iiwa arm both in simulation and on
hardware.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import FindResource

from drake.examples.manipulation_station.trajectory_planner import TrajectoryPlanner
from pydrake.math import (RigidTransform, RollPitchYaw, RotationMatrix)
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import (
    ApplySimulatorConfig, Simulator, SimulatorConfig)
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import (
    CollisionFilterDeclaration, GeometrySet, Meshcat, MeshcatVisualizer)
from pydrake.examples.manipulation_station import (
    ManipulationStation, ManipulationStationHardwareInterface)


# Environment parameters
sim_dt = 5e-3
table_height = 0.0
target_offset_z = 0.1
box_size = np.array([0.15, 0.15, 0.15])
box_mass = 1
box_mu = 1.0
contact_model = 'point'
contact_solver = 'sap'
initial_arm_pos = np.array(
    [0.32050248, 0.28234945, 0.33277261, 0.27744095, 0.26174226, 0.29629105, 0.30471719])
initial_box_pos = np.array([1, 0, 0, 0, 0.6, 0, 0.075])
desired_box_pos = np.array([1, 0, 0, 0, 1, 0, 0.075])


def AddTargetPosVisuals(plant, xyz_position, color=[.8, .1, .1, 1.0]):
    parser = Parser(plant)
    marker = parser.AddModelFromFile(FindResource("models/cross.sdf"))
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("cross", marker),
        RigidTransform(RollPitchYaw(0, 0, 0), np.array(xyz_position)))

# filter collisison between parent and child of each joint.


def add_collision_filters(scene_graph, plant):
    filter_manager = scene_graph.collision_filter_manager()
    body_pairs = [
        ["iiwa_link_1", "iiwa_link_2"],
        ["iiwa_link_2", "iiwa_link_3"],
        ["iiwa_link_3", "iiwa_link_4"],
        ["iiwa_link_4", "iiwa_link_5"],
        ["iiwa_link_5", "iiwa_link_6"],
        ["iiwa_link_6", "iiwa_link_7"],
    ]

    for pair in body_pairs:
        parent = plant.GetBodyByName(pair[0])
        child = plant.GetBodyByName(pair[1])

        set = GeometrySet(
            plant.GetCollisionGeometriesForBody(parent) +
            plant.GetCollisionGeometriesForBody(child))
        filter_manager.Apply(
            declaration=CollisionFilterDeclaration().ExcludeWithin(
                set))


def make_environment(meshcat=None, hardware=False, args=None):
    builder = DiagramBuilder()

    target_position = desired_box_pos[4:7]

    if hardware:
        camera_ids = []
        station = builder.AddSystem(ManipulationStationHardwareInterface(
            camera_ids, False, True))
        station.Connect(wait_for_cameras=False,
                        wait_for_wsg=False, wait_for_optitrack=False)
        controller_plant = station.get_controller_plant()
        plant = None
    else:
        station = builder.AddSystem(ManipulationStation(
            time_step=sim_dt, contact_model=contact_model, contact_solver=contact_solver))
        station.SetupCitoRlStation()

        station.AddManipulandFromFile(
            "drake/examples/manipulation_station/models/custom_box.sdf",
            RigidTransform(RotationMatrix.Identity(), np.zeros(3)), "box")

        controller_plant = station.get_controller_plant()
        plant = station.get_multibody_plant()

        AddTargetPosVisuals(plant, target_position)
        station.Finalize()

        if meshcat:
            geometry_query_port = station.GetOutputPort("geometry_query")
            meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
                builder=builder,
                query_object_port=geometry_query_port,
                meshcat=meshcat)

    diagram = builder.Build()

    return diagram, plant, controller_plant, station


def simulate_diagram(diagram, plant, controller_plant, station,
                     simulation_time, target_realtime_rate, hardware=False,
                     args=None, differential_ik=None):
    # Create context for the diagram
    diagram_context = diagram.CreateDefaultContext()

    # setup the simulator
    simulator_config = SimulatorConfig(
        target_realtime_rate=target_realtime_rate,
        publish_every_time_step=False)

    simulator = Simulator(diagram, diagram_context)

    ApplySimulatorConfig(config=simulator_config, simulator=simulator)
    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    if not hardware:
        plant_context = diagram.GetMutableSubsystemContext(plant,
                                                           diagram_context)
        print("Initial state variables: ",
              plant.GetPositionsAndVelocities(plant_context))

    context = simulator.get_mutable_context()
    context.SetTime(0)

    sim_t = 0
    if hardware:
        sim_t += 1e-6
        simulator.AdvanceTo(sim_t)
    else:
        # set the system pose to the prescribed values
        plant.SetPositions(plant_context, np.hstack(
            (initial_arm_pos, initial_box_pos)))

    # get the initial pose from the robot
    q0_arm = station.GetOutputPort("iiwa_position_measured").Eval(
        station_context)
    # get the box pose from the mo-cap
    q0_box = station.GetOutputPort(
        "optitrack_manipuland_pose").Eval(station_context)
    # set the initial pose of the system
    q0 = np.hstack((q0_arm, 1, 0, 0, 0, q0_box.translation()[0],
                    q0_box.translation()[1], box_size[2]/2))

    # plan a trajectory
    planner = TrajectoryPlanner(q0, desired_box_pos[4], args.preview)
    plan = planner.plan()

    # check for user permit
    user_permit = input("Do you want to run this trajectory? (y/N)")
    if user_permit != 'y':
        return 0

    # run a simulation executing the planned trajectory
    if not hardware:
        simulator.Initialize()
        input("\nPress Enter to run the simulation...")
        for _ in range(int(simulation_time/sim_dt)):
            sim_t += sim_dt
            q_cmd = plan.value(sim_t)
            station.GetInputPort("iiwa_position").FixValue(
                station_context, q_cmd)
            print(f"\tt: {sim_t}, qpos: {q_cmd.T}")
            simulator.AdvanceTo(sim_t)
        print("\nThe simulation has been completed")
        return 1

    # otherwise, command joint poses based on the time
    input("Press Enter to start the execution...\n\n")
    cur_time_s = -1
    start_time_us = station.GetOutputPort("iiwa_utime").Eval(station_context)
    while cur_time_s < 10:
        # evaluate the time from the start in s
        cur_time_s = (station.GetOutputPort("iiwa_utime").Eval(station_context) -
                      start_time_us) / 1e6
        # evaluate the corresponding joint pose command
        q_cmd = plan.value(cur_time_s)
        print(f"\ttime: {cur_time_s}, cmd: {q_cmd}")
    print("\nCompleted the execution")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meshcat", action="store_true",
        help="If set, visualize in meshcat. Use DrakeVisualizer otherwise")
    parser.add_argument(
        "--hardware", action="store_true",
        help="Use the ManipulationStationHardwareInterface instead of an "
             "in-process simulation.")
    parser.add_argument(
        "--preview", action="store_true",
        help="Preview the planned trajectory before execution.")
    args = parser.parse_args()

    if args.meshcat:
        meshcat_server = Meshcat()
        visualizer = meshcat_server
    else:
        visualizer = None

    input("Press Enter to continue...")

    diagram, plant, controller_plant, station = make_environment(
        meshcat=visualizer, hardware=args.hardware, args=args)

    simulate_diagram(
        diagram, plant, controller_plant, station,
        10.0, 1.0, hardware=args.hardware, args=args)
