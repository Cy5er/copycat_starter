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
    def __init__(self, skel_state: SkeletonState, skel_actors, axes_actors):
        """
        Initializes the application state.
        The application state is a canonical way to use the immediate mode GUI.
        Whatever change made to the GUI will probably require a change in the application state definition.

        Args:
            skel_state (SkeletonState): The state of the skeleton.
            skel_actors: List of actors representing the skeleton.
            axes_actors: List of actors representing the axes.
        """
        self.skel_state = skel_state
        self.skel_actors = skel_actors
        self.axes_actors = axes_actors
        self.mocap_file = COPYCAT_PATH
        self.mocap_dict = {"Default": []}
        self.mocap_idx = 0

        self.frame_idx = 0
        self.is_playing = False
        self.show_axes = False


# You may want to use these body names to get the index of the body part in the skeleton state.
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
    """
    Sets up and runs the GUI for the application.
    You will want to observe the patterns of immediate mode GUI programming to understand how the GUI is set up.

    Args:
        pl (ImguiPlotter): The ImGui plotter instance used for rendering.
        app_state (AppState): The state of the application including skeleton, axes, and mocap data.
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
            app_state.frame_idx = (
                0  # Reset frame index when a new mocap file is selected
            )
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
            app_state.is_playing = (
                True  # Set playing state to True when Play is clicked
            )
        imgui.same_line()
        if imgui.button("Pause"):
            app_state.is_playing = False  # Pause playback
        imgui.same_line()
        if imgui.button("Stop"):
            app_state.is_playing = False  # Stop playback and reset frame index
            app_state.frame_idx = 0

        # Automatic frame advancement when playing
        if app_state.is_playing:
            app_state.frame_idx += 1
            app_state.frame_idx %= (
                mocap_length  # Loop frame index within the mocap length
            )

        # Toggle visibility of axes in the scene
        changed, app_state.show_axes = imgui.checkbox("Show Axes", app_state.show_axes)
        imgui.end()

        # Set skeleton pose based on the mocap data at the current frame
        set_skel_pose(
            app_state.skel_state,
            app_state.skel_actors,
            app_state.axes_actors,
            app_state.show_axes,
        )

        # Sleep to maintain the target frame rate
        time_to_sleep = (1 / target_fps) - imgui_io.delta_time
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

    # Configure GUI behavior
    runner_params.callbacks.show_gui = gui
    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.no_default_window
    )

    # Start the GUI application loop
    immapp.run(runner_params=runner_params)


def main():
    # Initialize PyVista plotter and set a few parameters
    # Feel free to adjust these to your needs.
    pl = ImguiPlotter()
    pl.enable_shadows()  # Enable shadows for the 3D scene
    pl.add_axes()  # Add default axes for the scene
    # Set camera position, focus, and orientation
    pl.camera.position = (-5, -5, 2)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0, 0, 1)

    # PyVista works with meshes (intended to be static 3D data) and actors (manipulable objects corresponding to meshes)
    # Load skeleton and axes meshes from the MJCF file
    floor = pv.Plane(i_size=10, j_size=10)  # Create a floor plane
    skels = load_skel(MJCF_PATH)  # Load skeleton meshes
    axes = load_axes(MJCF_PATH)  # Load axes meshes

    # We have created meshes. Now we "add" them to the plotter via `.add_mesh()` to get actors
    pl.add_mesh(floor, show_edges=True, pbr=True, roughness=0.24, metallic=0.1)
    sk_actors = add_skel_meshes(pl, skels)  # multiple actors are returned
    ax_actors = add_axes_meshes(pl, axes)  # multiple actors are returned
    
    L_Shoulder_idx = body_names.index("L_Shoulder")
    R_Shoulder_idx = body_names.index("R_Shoulder")
    L_Elbow_idx = body_names.index("L_Elbow")
    R_Elbow_idx = body_names.index("R_Elbow")
    L_Hand_idx = body_names.index("L_Hand")
    R_Hand_idx = body_names.index("R_Hand")
    L_Wrist_idx = body_names.index("L_Wrist")
    R_Wrist_idx = body_names.index("R_Wrist")
    Spine_idx = body_names.index("Spine")
    Chest_idx = body_names.index("Chest")
    Neck_idx = body_names.index("Neck")
    L_Hip_idx = body_names.index("L_Hip")
    R_Hip_idx = body_names.index("R_Hip")
    L_Knee_idx = body_names.index("L_Knee")
    R_Knee_idx = body_names.index("R_Knee")
    L_Ankle_idx = body_names.index("L_Ankle")
    R_Ankle_idx = body_names.index("R_Ankle")

    # Set character pose to default: T-pose
    # Center the character root at the origin
    root_translation = torch.zeros(3)
    body_part_global_rotation = torch.zeros(24, 4)
    body_part_global_rotation[..., -1] = 1
    
    #Upsidedown T-pose
    #root_translation = torch.tensor([0.0, 0.0, 0.6772])
    #body_part_global_rotation[..., 0] = 1 
    
    # Y-pose
    # root_translation = torch.tensor([0.0,0.0,0.94])
    
    # body_part_global_rotation[L_Shoulder_idx] = torch.tensor([0.7, 0, 0, 0])
    # body_part_global_rotation[L_Elbow_idx] = torch.tensor([0.7, 0, 0, 0])
    # body_part_global_rotation[L_Wrist_idx] = torch.tensor([0.7, 0, 0, 0])
    # body_part_global_rotation[L_Hand_idx] = torch.tensor([0.7, 0, 0, 0])
    
    # body_part_global_rotation[R_Shoulder_idx] = torch.tensor([-0.7, 0, 0, 0])
    # body_part_global_rotation[R_Elbow_idx] = torch.tensor([-0.7, 0, 0, 0])
    # body_part_global_rotation[R_Wrist_idx] = torch.tensor([-0.7, 0, 0, 0])
    # body_part_global_rotation[R_Hand_idx] = torch.tensor([-0.7, 0, 0, 0])
    # body_part_global_rotation[..., -1] = 1
    
    #Fetal postion
    #root_translation = torch.tensor([0.0, 0.0, 0.26])
    #body_part_global_rotation[Spine_idx] = torch.tensor([1, 1, 1, 1])
    #body_part_global_rotation[R_Hip_idx] = torch.tensor([1, 0, -1, 0])
    #body_part_global_rotation[L_Hip_idx] = torch.tensor([1, 0, -1, 0])
    
    #body_part_global_rotation[R_Shoulder_idx] = torch.tensor([1, -1, 0, 0])
    #body_part_global_rotation[R_Elbow_idx] = torch.tensor([1, -1, -1, -1])
    #body_part_global_rotation[L_Shoulder_idx] = torch.tensor([0.7, 0, 0, 0])
    #body_part_global_rotation[..., 0] = 1

    # `poselib` has helpful classes called `SkeletonTree` and `SkeletonState` that handle forward kinematics
    sk_tree = SkeletonTree.from_mjcf(MJCF_PATH)
    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree, body_part_global_rotation, root_translation, is_local=False
    )
    set_skel_pose(
        sk_state, sk_actors, ax_actors, show_axes=True
    )  # Sets the pose of each individual actor.

    # Run the GUI
    app_state = AppState(sk_state, sk_actors, ax_actors)
    app_state.mocap_dict = joblib.load(app_state.mocap_file)
    setup_and_run_gui(pl, app_state)


if __name__ == "__main__":
    main()