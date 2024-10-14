from typing import List

import numpy as np
import pyvista as pv
import torch
from imgui_bundle import imgui, immapp, hello_imgui
from pyvista_imgui import ImguiPlotter
from scipy.spatial.transform import Rotation
from torch import nn

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from utils import set_skel_pose, add_3p_meshes, load_skel, load_axes, add_skel_meshes, add_axes_meshes
from visual_data import XMLVisualDataContainer

MJCF_PATH = "assets/my_smpl_humanoid.xml"

# Define the model for predicting the full-body pose from 3-point tracking (positions + rotations)
class PoseEstimationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PoseEstimationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Create the model instance and load pre-trained weights (for simplicity, initialized randomly here)
model = PoseEstimationModel(input_size=18, hidden_size=256, output_size=96 + 3)  # 18 input (3 points x (3D position + 3D rotation)), 96 output for pose, 3 for root translation
# You would load your trained model weights here
# model.load_state_dict(torch.load('path_to_trained_model.pth'))
model.eval()

class AppState:
    def __init__(
        self, skel_state: SkeletonState, skel_actors, axes_actors, three_p_actors, model
    ):
        self.skel_state = skel_state
        self.skel_actors = skel_actors
        self.axes_actors = axes_actors
        self.three_p_actors = three_p_actors
        self.model = model  # Integrate the model into the AppState

        self.three_p_pos: np.ndarray = np.zeros((3, 3))
        self.three_p_rot: np.ndarray = np.zeros((3, 3), dtype=int)

    # Function to use the model and get the predicted pose
    def predict_pose(self):
        # Create input for the model using both positions and rotations
        input_data = np.concatenate([
            self.three_p_pos.flatten(),  # Positions for Head, Left Hand, Right Hand (3x3)
            self.three_p_rot.flatten()   # Rotations for Head, Left Hand, Right Hand (3x3)
        ])
        input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add batch dimension
        
        # Use the model to predict the full-body pose
        predicted_pose = self.model(input_data).detach().numpy().flatten()
        return predicted_pose

body_names = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]

def setup_and_run_gui(pl: ImguiPlotter, app_state: AppState):
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Viewer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)

    def gui():
        imgui_io = imgui.get_io()
        hello_imgui.apply_theme(hello_imgui.ImGuiTheme_.imgui_colors_dark)

        viewport_size = imgui.get_window_viewport().size

        # PyVista portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(viewport_size.x // 2, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "ImguiPlotter",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus
            | imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move,
        )
        # render the plotter's contents here
        pl.render_imgui()
        imgui.end()

        # GUI portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "Controls",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move,
        )

        # GUI controls for positions and rotations
        imgui.text("Head")
        changed, app_state.three_p_pos[0, 0] = imgui.slider_float(
            "Head X Pos", app_state.three_p_pos[0, 0], -1, 1
        )
        changed, app_state.three_p_pos[0, 1] = imgui.slider_float(
            "Head Y Pos", app_state.three_p_pos[0, 1], -1, 1
        )
        changed, app_state.three_p_pos[0, 2] = imgui.slider_float(
            "Head Z Pos", app_state.three_p_pos[0, 2], 0, 2
        )
        changed, app_state.three_p_rot[0, 0] = imgui.slider_int(
            "Head X Rot", app_state.three_p_rot[0, 0], -180, 180
        )
        changed, app_state.three_p_rot[0, 1] = imgui.slider_int(
            "Head Y Rot", app_state.three_p_rot[0, 1], -180, 180
        )
        changed, app_state.three_p_rot[0, 2] = imgui.slider_int(
            "Head Z Rot", app_state.three_p_rot[0, 2], -180, 180
        )

        imgui.text("Left Hand")
        changed, app_state.three_p_pos[1, 0] = imgui.slider_float(
            "Left Hand X Pos", app_state.three_p_pos[1, 0], -1, 1
        )
        changed, app_state.three_p_pos[1, 1] = imgui.slider_float(
            "Left Hand Y Pos", app_state.three_p_pos[1, 1], -1, 1
        )
        changed, app_state.three_p_pos[1, 2] = imgui.slider_float(
            "Left Hand Z Pos", app_state.three_p_pos[1, 2], 0, 2
        )
        changed, app_state.three_p_rot[1, 0] = imgui.slider_int(
            "Left Hand X Rot", app_state.three_p_rot[1, 0], -180, 180
        )
        changed, app_state.three_p_rot[1, 1] = imgui.slider_int(
            "Left Hand Y Rot", app_state.three_p_rot[1, 1], -180, 180
        )
        changed, app_state.three_p_rot[1, 2] = imgui.slider_int(
            "Left Hand Z Rot", app_state.three_p_rot[1, 2], -180, 180
        )

        imgui.text("Right Hand")
        changed, app_state.three_p_pos[2, 0] = imgui.slider_float(
            "Right Hand X Pos", app_state.three_p_pos[2, 0], -1, 1
        )
        changed, app_state.three_p_pos[2, 1] = imgui.slider_float(
            "Right Hand Y Pos", app_state.three_p_pos[2, 1], -1, 1
        )
        changed, app_state.three_p_pos[2, 2] = imgui.slider_float(
            "Right Hand Z Pos", app_state.three_p_pos[2, 2], 0, 2
        )
        changed, app_state.three_p_rot[2, 0] = imgui.slider_int(
            "Right Hand X Rot", app_state.three_p_rot[2, 0], -180, 180
        )
        changed, app_state.three_p_rot[2, 1] = imgui.slider_int(
            "Right Hand Y Rot", app_state.three_p_rot[2, 1], -180, 180
        )
        changed, app_state.three_p_rot[2, 2] = imgui.slider_int(
            "Right Hand Z Rot", app_state.three_p_rot[2, 2], -180, 180
        )

        imgui.end()

        # Predict pose from model and update the skeleton state
        predicted_pose = app_state.predict_pose()
        predicted_rotation = torch.tensor(predicted_pose[:96]).reshape(24, 4)
        predicted_translation = torch.tensor(predicted_pose[96:])

        # Update skeleton state
        app_state.skel_state = SkeletonState.from_rotation_and_root_translation(
            app_state.skel_state.skeleton_tree,
            predicted_rotation,
            predicted_translation,
            is_local=False
        )

        set_skel_pose(app_state.skel_state, app_state.skel_actors, app_state.axes_actors, show_axes=False)


    runner_params.callbacks.show_gui = gui
    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.no_default_window
    )
    immapp.run(runner_params=runner_params)


def main():
    pl = ImguiPlotter()
    pl.enable_shadows()
    pl.add_axes()
    pl.camera.position = (-5, -5, 2)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0, 0, 1)

    # Initialize meshes
    floor = pv.Plane(i_size=10, j_size=10)
    skels = load_skel(MJCF_PATH)
    axes = load_axes(MJCF_PATH)

    # Register meshes, get actors for object manipulation
    pl.add_mesh(floor, show_edges=True, pbr=True, roughness=0.24, metallic=0.1)
    sk_actors = add_skel_meshes(pl, skels)
    ax_actors = add_axes_meshes(pl, axes)

    # Set character pose to default
    root_translation = torch.zeros(3)
    body_part_global_rotation = torch.zeros(24, 4)
    body_part_global_rotation[..., -1] = 1

    sk_tree = SkeletonTree.from_mjcf(MJCF_PATH)
    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree, body_part_global_rotation, root_translation, is_local=False
    )
    set_skel_pose(sk_state, sk_actors, ax_actors, show_axes=False)

    three_p_actors = add_3p_meshes(pl)

    # Run the GUI
    app_state = AppState(sk_state, sk_actors, ax_actors, three_p_actors, model)
    setup_and_run_gui(pl, app_state)


if __name__ == "__main__":
    main()