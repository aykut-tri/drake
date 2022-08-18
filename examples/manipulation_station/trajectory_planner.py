import numpy as np
import time

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import (MeshcatVisualizer, SceneGraph, StartMeshcat)
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph, MultibodyPlant)
from pydrake.solvers import (BoundingBoxConstraint, IpoptSolver, SnoptSolver)
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.trajectory_optimization import DirectCollocation
from pydrake.trajectories import PiecewisePolynomial


class TrajectoryPlanner():
    def __init__(self, initial_pose, desired_box_x, preview=False):
        self.q0 = initial_pose
        self.xd = desired_box_x
        self.preview = preview
        # Create the environment
        builder = DiagramBuilder()
        self.plant, self.scene = AddMultibodyPlantSceneGraph(builder, 5e-2)
        self.build_environment(add_manipuland=False)
        # Add a meshcat visualizer
        MeshcatVisualizer.AddToBuilder(builder=builder, scene_graph=self.scene,
                                       meshcat=StartMeshcat())
        # Build the diagram
        self.diagram = builder.Build()
        # Create contexts
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(
            self.diagram_context)

        # Visualize the initial pose
        self.diagram_context.SetDiscreteState(
            np.hstack((self.q0[:7], np.zeros(self.plant.num_velocities()))))
        self.diagram.Publish(self.diagram_context)

        # # Create ports and objects for collision computations
        # self.query_port = self.plant.get_geometry_query_input_port()
        # self.query_object = self.query_port.Eval(self.plant_context)
        # self.inspector = self.query_object.inspector()

        # # Specify the indices of the desired contact pairs
        # contact_candidate_ids = [0, 3, 5, 7, 9]
        # # Get the contact candidates
        # self.contact_candidates = []
        # self.X_AGa = []
        # self.X_BGb = []
        # self.pair_id_iiwa_box = None
        # self.get_contact_candidates(contact_candidate_ids=contact_candidate_ids,
        #                             print_all=False)

        # # Create AutoDiffXd correspondences
        # diagram_ad = diagram.ToAutoDiffXd()
        # diagram_context_ad = diagram_ad.CreateDefaultContext()
        # self.plant_ad = diagram_ad.GetSubsystemByName(self.plant.get_name())
        # self.plant_context_ad = self.plant_ad.GetMyContextFromRoot(diagram_context_ad)

    def plan(self):
        # Plan a trajectory to the desired joint pose
        time, states = self.plan_to_joint_pose(
            q_goal=np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, 0]), num_time_samples=31)

        # Preview the planned trajectory
        if self.preview:
            input("\nPress Enter to preview the planned trajectory...\n")
            for i, t in enumerate(time):
                print(f"t = {t}")
                print(f"\tq = {states[:7, i]}")
                print(f"\tv = {states[7:, i]}")
                self.diagram_context.SetDiscreteState(states[:, i])
                self.diagram.Publish(self.diagram_context)
                if t != time[-1]:
                    input("Press Enter to continue...")
            print("The preview is completed\n")

        # Smoothen and return the position trajectory
        plan = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            time, states[:7, :], np.zeros(7), np.zeros(7))
        # t_int = np.linspace(0, time[-1], 100)
        # x_int = plan.vector_values(t_int)
        return plan

    def plan_to_joint_pose(self, q_goal, num_time_samples=11):
        # Build a continuous-time plant and a scene
        plant = MultibodyPlant(0)
        scene_graph = SceneGraph()
        plant.RegisterAsSourceForSceneGraph(scene_graph)
        # Load the iiwa model
        model_file = FindResourceOrThrow(
            "drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf"
        )
        Parser(plant).AddModelFromFile(model_file)
        # Attach the base to the world frame and complete the plant definition
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("iiwa_link_0"))
        plant.Finalize()
        # Create a context for the plant
        context = plant.CreateDefaultContext()

        # Set up the collocation problem
        dircol = DirectCollocation(
            system=plant, context=context,
            num_time_samples=num_time_samples, minimum_timestep=2e-2, maximum_timestep=1e-1,
            input_port_index=plant.get_actuation_input_port().get_index())

        # Get the associated mathematical program
        prog = dircol.prog()
        x = dircol.state()
        u = dircol.input()

        # # Enforce equally-divided time segments
        # dircol.AddEqualTimeIntervalsConstraints()

        # Specify the task
        x0 = np.hstack((self.q0[:7], np.zeros(7)))
        xf = np.hstack((q_goal, np.zeros(7)))
        # Add the relevant constraints
        prog.AddBoundingBoxConstraint(x0, x0, dircol.initial_state())
        prog.AddBoundingBoxConstraint(xf, xf, dircol.final_state())


        # Add joint position, velocity, and effort constraints
        dircol.AddConstraintToAllKnotPoints(BoundingBoxConstraint(0.8*plant.GetPositionLowerLimits(),
                                                                  0.8*plant.GetPositionUpperLimits()),
                                            x[:7])
        dircol.AddConstraintToAllKnotPoints(BoundingBoxConstraint(0.7*plant.GetVelocityLowerLimits(),
                                                                  0.7*plant.GetVelocityUpperLimits()),
                                            x[7:])
        dircol.AddConstraintToAllKnotPoints(BoundingBoxConstraint(0.7*plant.GetEffortLowerLimits(),
                                                                  0.7*plant.GetEffortUpperLimits()),
                                            u)

        # Penalize the total time
        dircol.AddFinalCost(dircol.time())

        # Penalize the velocities
        w_vel = 0
        if w_vel > 0:
            dircol.AddRunningCost(w_vel * x[7:].dot(x[7:]))

        # Penalize the effort
        w_tau = 1
        if w_tau > 0:
            dircol.AddRunningCost(w_tau * u.dot(u))

        # Set an initial guess by interpolating between the initial and final states
        initial_x_trajectory = PiecewisePolynomial.FirstOrderHold(
            [0., 2.], np.column_stack((x0, xf)))
        dircol.SetInitialTrajectory(
            PiecewisePolynomial(), initial_x_trajectory)

        # Solve the program
        solver = SnoptSolver()
        print("\nRunning trajectory optimization...\n")
        t_start = time.time()
        result = solver.Solve(prog)
        # Print solver details
        print(f"\tSolver type: {result.get_solver_id().name()}")
        print(f"\tSolver took {time.time() - t_start} s")
        print(f"\tSuccess: {result.is_success()}")
        print(f"\tOptimal cost: {result.get_optimal_cost()}")

        # Get the solution
        t = dircol.GetSampleTimes(result)
        x = dircol.GetStateSamples(result)
        return (t, x)

    def build_environment(self, add_manipuland=False):
        # Add a table (i.e., a box) and fix it to the world
        sdf_file = FindResourceOrThrow(
            "drake/examples/manipulation_station/models/floor.sdf")
        Parser(self.plant, self.scene).AddModelFromFile(sdf_file)
        self.plant.WeldFrames(frame_on_parent_F=self.plant.world_frame(),
                              frame_on_child_M=self.plant.GetFrameByName(
                                  "floor"),
                              X_FM=RigidTransform(p=[1.5, 0.0, -2.5e-2]))
        # Add an iiwa and fix its base to the world
        sdf_file = FindResourceOrThrow(
            "drake/examples/manipulation_station/models/iiwa14_point_end_effector.sdf")
        Parser(self.plant, self.scene).AddModelFromFile(sdf_file)
        self.plant.WeldFrames(frame_on_parent_F=self.plant.world_frame(),
                              frame_on_child_M=self.plant.GetFrameByName(
                                  "iiwa_link_0"),
                              X_FM=RigidTransform(p=[0.0, 0.0, 0.0]))
        # Add the manipuland
        if add_manipuland:
            sdf_file = FindResourceOrThrow(
                "drake/examples/manipulation_station/models/custom_box.sdf")
            Parser(self.plant, self.scene).AddModelFromFile(sdf_file)
        # Finalize the plant
        self.plant.Finalize()

    def get_contact_candidates(self, contact_candidate_ids, print_all=False):
        # Get all contact pairs
        all_contact_pairs = self.inspector.GetCollisionCandidates()
        if print_all:
            print("All collision pairings:")
            for pair_id, pair in enumerate(all_contact_pairs):
                print(f"\tPair {pair_id}")
                print(f"\t\t{pair[0]}: {self.inspector.GetName(pair[0])}")
                print(f"\t\t{pair[1]}: {self.inspector.GetName(pair[1])}")

        # Get the desired contact pairs
        self.contact_candidates = []
        for pair_id, pair in enumerate(all_contact_pairs):
            if pair_id in contact_candidate_ids:
                self.contact_candidates.append(pair)

        # Process the resulting contact candidates
        print("Contact candidates for planning:")
        for pair_id, pair in enumerate(self.contact_candidates):
            print(f"\tPair {pair_id}")
            print(f"\t\t{pair[0]}: {self.inspector.GetName(pair[0])}")
            print(f"\t\t{pair[1]}: {self.inspector.GetName(pair[1])}")
            # Get the collision geometry's pose in the body frame
            self.X_AGa.append(self.inspector.GetPoseInFrame(pair[0]))
            self.X_BGb.append(self.inspector.GetPoseInFrame(pair[1]))
            # Get the pair ID for the iiwa-box contact
            if (self.inspector.GetName(pair[0]) == "custom_box::push_point_collision" or
                    self.inspector.GetName(pair[1]) == "custom_box::push_point_collision"):
                self.pair_id_iiwa_box = pair_id
        print(f"Pair ID for the iiwa-box contact: {self.pair_id_iiwa_box}")
