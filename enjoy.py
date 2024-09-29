from typing import List

import numpy as np
import pyvista as pv
import torch
from imgui_bundle import imgui, immapp, hello_imgui
from pyvista_imgui import ImguiPlotter
from scipy.spatial.transform import Rotation

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from utils import set_skel_pose, add_3p_meshes, load_skel, load_axes, add_skel_meshes, add_axes_meshes
from visual_data import XMLVisualDataContainer

MJCF_PATH = "assets/my_smpl_humanoid.xml"


class AppState:
    def __init__(
        self, skel_state: SkeletonState, skel_actors, axes_actors, three_p_actors
    ):
        self.skel_state = skel_state
        self.skel_actors = skel_actors
        self.axes_actors = axes_actors
        self.three_p_actors = three_p_actors

        self.three_p_pos: np.ndarray = np.zeros((3, 3))
        self.three_p_rot: np.ndarray = np.zeros((3, 3), dtype=int)


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

        for i in range(3):
            m = np.eye(4)
            m[:3, 3] = app_state.three_p_pos[i]
            m[:3, :3] = Rotation.from_euler(
                "xyz", app_state.three_p_rot[i], degrees=True
            ).as_matrix()
            app_state.three_p_actors[i].user_matrix = m

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
    # Center the character root at the origin
    root_translation = torch.zeros(3)
    # Set global rotation as unit quaternion
    body_part_global_rotation = torch.zeros(24, 4)
    body_part_global_rotation[..., -1] = 1

    # `poselib` handles the forward kinematics
    sk_tree = SkeletonTree.from_mjcf(MJCF_PATH)
    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree, body_part_global_rotation, root_translation, is_local=False
    )
    set_skel_pose(sk_state, sk_actors, ax_actors, show_axes=False)

    three_p_actors = add_3p_meshes(pl)

    # Run the GUI
    app_state = AppState(sk_state, sk_actors, ax_actors, three_p_actors)
    setup_and_run_gui(pl, app_state)


if __name__ == "__main__":
    main()
