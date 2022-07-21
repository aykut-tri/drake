import argparse
import webbrowser

import numpy as np

from pydrake.examples import (
    ManipulationStation, ManipulationStationHardwareInterface,
    SchunkCollisionModel)
from pydrake.geometry import DrakeVisualizer, Meshcat, MeshcatVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target_realtime_rate", type=float, default=1.0,
        help="Desired rate relative to real time.  See documentation for "
             "Simulator::set_target_realtime_rate() for details.")
    parser.add_argument(
        "--duration", type=float, default=np.inf,
        help="Desired duration of the simulation in seconds.")
    parser.add_argument(
        "--hardware", action='store_true',
        help="Use the ManipulationStationHardwareInterface instead of an "
             "in-process simulation.")
    parser.add_argument(
        "--meshcat", action="store_true", default=False,
        help="Enable visualization with meshcat.")
    parser.add_argument(
        "--planar", action="store_true", default=False,
        help="Enable planar view.")
    parser.add_argument(
        "-w", "--open-window", dest="browser_new",
        action="store_const", const=1, default=None,
        help="Open the MeshCat display in a new browser window.")
    args = parser.parse_args()

    builder = DiagramBuilder()

    if args.hardware:
        station = builder.AddSystem(ManipulationStationHardwareInterface())
        station.Connect(wait_for_cameras=False)
    else:
        station = builder.AddSystem(ManipulationStation())

        # Initializes the chosen station type.
        schunk_model = SchunkCollisionModel.kBoxPlusFingertipSpheres
        station.SetupManipulationClassStation(
            schunk_model=schunk_model)

        station.Finalize()
        query_port = station.GetOutputPort("query_object")

        DrakeVisualizer.AddToBuilder(builder, query_port)
        if args.meshcat:
            meshcat = Meshcat()
            meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
                builder=builder,
                query_object_port=query_port,
                meshcat=meshcat)

            if args.planar:
                meshcat.Set2dRenderMode()

            if args.browser_new is not None:
                url = meshcat.web_url()
                webbrowser.open(url=url, new=args.browser_new)

    # Create a diagram and a corresponding simulation.
    diagram = builder.Build()
    simulator = Simulator(diagram)

    # This is important to avoid duplicate publishes to the hardware interface:
    simulator.set_publish_every_time_step(False)

    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    # Set the joint pose command.
    station.GetInputPort("iiwa_position").FixValue(
        station_context, np.array([0, 0.75, 0, -1.9, 0, -1, 0]))

    # Fix the gripper at open position.
    station.GetInputPort("wsg_position").FixValue(
        station_context, 0.1)

    # Set the feed-forward toque to zero.
    station.GetInputPort("iiwa_feedforward_torque").FixValue(
        station_context, np.zeros(7))

    # If the diagram is only the hardware interface, then we must advance it a
    # little bit so that first LCM messages get processed. A simulated plant is
    # already publishing correct positions even without advancing, and indeed
    # we must not advance a simulated plant until the sliders and filters have
    # been initialized to match the plant.
    if args.hardware:
        simulator.AdvanceTo(1e-6)

    station.GetOutputPort("iiwa_position_measured").Eval(station_context)

    simulator.set_target_realtime_rate(args.target_realtime_rate)

    simulator.AdvanceTo(args.duration)


if __name__ == '__main__':
    main()
