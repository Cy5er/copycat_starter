from imgui_bundle import imgui, immapp, hello_imgui
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from pyvista_imgui import ImguiPlotter
from utils import set_skel_pose, load_skel, load_axes, add_skel_meshes, add_axes_meshes
import joblib
import pyvista as pv
import time
import torch

MJCF_PATH = "assets/my_smpl_humanoid.xml"
COPYCAT_PATH = "data/amass_copycat_take5_train_medium.pkl"


class AppState:
    def __init__(self, skel_tree: SkeletonTree, skel_state: SkeletonState, skel_actors, axes_actors):
        """
        Initializes the application state.
        """
        self.skel_tree = skel_tree
        self.skel_state = skel_state
        self.skel_actors = skel_actors
        self.axes_actors = axes_actors
        self.mocap_file = COPYCAT_PATH
        self.mocap_dict = {"Default": []}  # Default mocap data dict, will load actual mocap below
        self.mocap_idx = 0
        self.frame_idx = 0
        self.is_playing = False
        self.show_axes = False
        self.mocap_length = 0  # Track the total frames for the mocap
        self.last_frame_time = time.time()  # To track time for FPS control

    def load_mocap_data(self):
        """Load mocap data from the specified file and update the dictionary."""
        self.mocap_dict = joblib.load(self.mocap_file)
        mocap_name = list(self.mocap_dict.keys())[self.mocap_idx]
        self.mocap_length = self.mocap_dict[mocap_name]['trans_orig'].shape[0]  # Update total frame count


def setup_and_run_gui(pl: ImguiPlotter, app_state: AppState):
    """
    Sets up and runs the GUI for the application.
    """
    # Initialize window properties
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Viewer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)

    target_fps = 30  # Set the target frame rate for the GUI

    def gui():
        imgui_io = imgui.get_io()  # Get the IO for managing frame timing and user input
        hello_imgui.apply_theme(hello_imgui.ImGuiTheme_.imgui_colors_dark)  # Set theme

        # Determine viewport size dynamically based on window dimensions
        viewport_size = imgui.get_window_viewport().size

        # PyVista rendering window (right half of the screen)
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
        # Render PyVista content
        pl.render_imgui()
        imgui.end()

        # Control panel window (left half of the screen)
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "Controls",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move,
        )

        # Mocap file and frame controls
        imgui.text("Mocap")

        # Display mocap file path
        imgui.input_text("CopycatFile", app_state.mocap_file, 0)

        # Mocap selection slider to pick a different mocap file
        changed, app_state.mocap_idx = imgui.slider_int(
            "MocapIdx", app_state.mocap_idx, 0, len(app_state.mocap_dict) - 1
        )
        if changed:
            app_state.frame_idx = 0  # Reset frame index when a new mocap file is selected
            app_state.load_mocap_data()

        app_state.mocap_name = list(app_state.mocap_dict.keys())[app_state.mocap_idx]

        # Display mocap name (read-only)
        imgui.input_text(
            "MocapName",
            app_state.mocap_name,
            imgui.InputTextFlags_.read_only,
        )

        # Mocap frame slider to navigate frames within the selected mocap
        mocap = app_state.mocap_dict[app_state.mocap_name]
        mocap_length = mocap["trans_orig"].shape[0]
        changed, app_state.frame_idx = imgui.slider_int(
            "Frame", app_state.frame_idx, 0, mocap_length - 1
        )

        # Playback controls (Play, Pause, Stop)
        if imgui.button("Play"):
            app_state.is_playing = True  # Set playing state to True when Play is clicked
        imgui.same_line()
        if imgui.button("Pause"):
            app_state.is_playing = False  # Pause playback
        imgui.same_line()
        if imgui.button("Stop"):
            app_state.is_playing = False  # Stop playback and reset frame index
            app_state.frame_idx = 0

        # Automatic frame advancement when playing
        current_time = time.time()
        if app_state.is_playing and current_time - app_state.last_frame_time >= 1 / target_fps:
            app_state.frame_idx += 1
            app_state.frame_idx %= mocap_length  # Loop frame index within the mocap length
            app_state.last_frame_time = current_time  # Update the last frame time

        # Toggle visibility of axes in the scene
        changed, app_state.show_axes = imgui.checkbox("Show Axes", app_state.show_axes)
        imgui.end()

        # Apply mocap data to skeleton pose
        translation = torch.as_tensor(mocap["trans_orig"][app_state.frame_idx])
        rotation = torch.as_tensor(mocap["pose_quat_global"][app_state.frame_idx])

        # Use the skel_tree from app_state instead of app_state.skel_state.tree
        app_state.skel_state = SkeletonState.from_rotation_and_root_translation(
            app_state.skel_tree, rotation, translation, is_local=False
        )

        # Apply the pose to the skeleton actors
        set_skel_pose(
            app_state.skel_state, app_state.skel_actors, app_state.axes_actors, app_state.show_axes
        )

    # Configure GUI behavior
    runner_params.callbacks.show_gui = gui
    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.no_default_window
    )

    # Start the GUI application loop
    immapp.run(runner_params=runner_params)


def main():
    # Initialize PyVista plotter and set a few parameters
    pl = ImguiPlotter()
    pl.enable_shadows()  # Enable shadows for the 3D scene
    pl.add_axes()  # Add default axes for the scene
    pl.camera.position = (-5, -5, 2)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0, 0, 1)

    # Load skeleton and axes meshes from the MJCF file
    floor = pv.Plane(i_size=10, j_size=10)  # Create a floor plane
    skels = load_skel(MJCF_PATH)  # Load skeleton meshes
    axes = load_axes(MJCF_PATH)  # Load axes meshes

    # Add floor and skeleton meshes to the plotter
    pl.add_mesh(floor, show_edges=True, pbr=True, roughness=0.24, metallic=0.1)
    sk_actors = add_skel_meshes(pl, skels)  # Add skeleton actors
    ax_actors = add_axes_meshes(pl, axes)  # Add axes actors

    # Initialize the skeleton to the T-pose
    root_translation = torch.zeros(3)
    body_part_global_rotation = torch.zeros(24, 4)
    body_part_global_rotation[..., -1] = 1
    sk_tree = SkeletonTree.from_mjcf(MJCF_PATH)
    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree, body_part_global_rotation, root_translation, is_local=False
    )

    # Initialize the app state with the skeleton actors and load mocap data
    app_state = AppState(sk_tree, sk_state, sk_actors, ax_actors)
    app_state.load_mocap_data()

    # Run the GUI
    setup_and_run_gui(pl, app_state)


if __name__ == "__main__":
    main()