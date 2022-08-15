import numpy as np

from pydrake.geometry import (MeshcatVisualizer, StartMeshcat)
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from utils import FindResource


class TrajectoryPlanner():
    def __init__(self, initial_pose, desired_box_x):
        self.q0 = initial_pose
        self.xd = desired_box_x
        # Create the environment
        builder = DiagramBuilder()
        self.plant, self.scene = AddMultibodyPlantSceneGraph(builder, 5e-2)
        self.build_environment()
        # Add visualizer
        MeshcatVisualizer.AddToBuilder(builder=builder, scene_graph=self.scene,
                                       meshcat=StartMeshcat())
        # Build
        diagram = builder.Build()
        # Create contexts
        diagram_context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(diagram_context)

        # Publish the initial pose
        diagram_context.SetDiscreteState(
            np.hstack((self.q0, np.zeros(self.plant.num_velocities()))))
        diagram.Publish(diagram_context)

        # Create ports and objects for collision computations
        self.query_port = self.plant.get_geometry_query_input_port()
        self.query_object = self.query_port.Eval(self.plant_context)
        self.inspector = self.query_object.inspector()

        # Specify the indices of the desired contact pairs
        contact_candidate_ids = [0, 3, 5, 7, 9]
        # Get the contact candidates
        self.contact_candidates = []
        self.X_AGa = []
        self.X_BGb = []
        self.pair_id_iiwa_box = None
        self.get_contact_candidates(contact_candidate_ids=contact_candidate_ids,
                                    print_all=False)

        # Create AutoDiffXd correspondences
        diagram_ad = diagram.ToAutoDiffXd()
        diagram_context_ad = diagram_ad.CreateDefaultContext()
        self.plant_ad = diagram_ad.GetSubsystemByName(self.plant.get_name())
        self.plant_context_ad = self.plant_ad.GetMyContextFromRoot(diagram_context_ad)


    def build_environment(self):
        # Add an iiwa and fix its base to the world
        Parser(self.plant, self.scene).AddModelFromFile(
            FindResource("models/iiwa14_point_end_effector.sdf"))
        self.plant.WeldFrames(frame_on_parent_F=self.plant.world_frame(),
                              frame_on_child_M=self.plant.GetFrameByName("iiwa_link_0"),
                              X_FM=RigidTransform(p=[0.0, 0.0, 0.0]))
        # Add a table (i.e., a box) and fix it to the world
        Parser(self.plant, self.scene).AddModelFromFile(FindResource("models/floor.sdf"))
        self.plant.WeldFrames(frame_on_parent_F=self.plant.world_frame(),
                              frame_on_child_M=self.plant.GetFrameByName("floor"),
                              X_FM=RigidTransform(p=[0.0, 0.0, 0.0]))
        # Add the manipuland
        Parser(self.plant, self.scene).AddModelFromFile(FindResource("models/custom_box.sdf"))
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

    def create_mathematical_program(self):
