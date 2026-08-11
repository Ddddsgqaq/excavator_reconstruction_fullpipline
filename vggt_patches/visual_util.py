# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import trimesh
import gradio as gr
import numpy as np
import matplotlib
from scipy.spatial.transform import Rotation
import copy
import cv2
import os
import requests


def predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    target_dir=None,
    prediction_mode="Predicted Pointmap",
    semantic_masks=None,
    enable_sam=False,
    semantic_filter_ids=None,
    semantic_filter_mode="delete",
) -> trimesh.Scene:
    """
    Converts VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)
        filter_by_frames (str): Frame filter specification (default: "all")
        mask_black_bg (bool): Mask out black background pixels (default: False)
        mask_white_bg (bool): Mask out white background pixels (default: False)
        show_cam (bool): Include camera visualization (default: True)
        mask_sky (bool): Apply sky segmentation mask (default: False)
        target_dir (str): Output directory for intermediate files (default: None)
        prediction_mode (str): Prediction mode selector (default: "Predicted Pointmap")

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10.0

    print("Building GLB scene")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]  # No batch dimension to remove
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            print("Warning: world_points not found in predictions, falling back to depth-based points")
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_matrices = predictions["extrinsic"]

    if mask_sky:
        if target_dir is not None:
            import onnxruntime

            skyseg_session = None
            target_dir_images = target_dir + "/images"
            image_list = sorted(os.listdir(target_dir_images))
            sky_mask_list = []

            # Get the shape of pred_world_points_conf to match
            S, H, W = (
                pred_world_points_conf.shape
                if hasattr(pred_world_points_conf, "shape")
                else (len(images), images.shape[1], images.shape[2])
            )

            # Download skyseg.onnx if it doesn't exist
            if not os.path.exists("skyseg.onnx"):
                print("Downloading skyseg.onnx...")
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
                )

            for i, image_name in enumerate(image_list):
                image_filepath = os.path.join(target_dir_images, image_name)
                mask_filepath = os.path.join(target_dir, "sky_masks", image_name)

                # Check if mask already exists
                if os.path.exists(mask_filepath):
                    # Load existing mask
                    sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                else:
                    # Generate new mask
                    if skyseg_session is None:
                        skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
                    sky_mask = segment_sky(image_filepath, skyseg_session, mask_filepath)

                # Resize mask to match H×W if needed
                if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                    sky_mask = cv2.resize(sky_mask, (W, H))

                sky_mask_list.append(sky_mask)

            # Convert list to numpy array with shape S×H×W
            sky_mask_array = np.array(sky_mask_list)

            # Apply sky mask to confidence scores
            sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
            pred_world_points_conf = pred_world_points_conf * sky_mask_binary

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]
        # Also slice semantic_masks if present
        if enable_sam and semantic_masks is not None:
            if semantic_masks.shape[0] > selected_frame_idx:
                semantic_masks = semantic_masks[selected_frame_idx][None]
            else:
                print(f"Warning: selected_frame_idx {selected_frame_idx} out of range for semantic_masks")
                semantic_masks = None

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    # Generate semantic colors BEFORE any filtering if SAM is enabled
    semantic_colors_full = None
    if enable_sam and semantic_masks is not None:
        print("Generating semantic colors...")
        print(f"Semantic masks shape: {semantic_masks.shape}")
        print(f"Images shape after selection: {images.shape}")
        print(f"Colors RGB shape: {colors_rgb.shape}")

        # Generate semantic colors - semantic_masks should now match images
        semantic_colors_full = get_semantic_colors(semantic_masks, selected_frame_idx=None)
        print(f"Semantic colors shape: {semantic_colors_full.shape}")

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres)

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        # Filter out white background pixels (RGB values close to white)
        # Consider pixels white if all RGB values are above 240
        white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask

    if semantic_filter_ids and semantic_masks is not None:
        sem_flat = semantic_masks.reshape(-1)
        sem_class_mask = np.isin(sem_flat, semantic_filter_ids)
        if semantic_filter_mode == "delete":
            conf_mask = conf_mask & ~sem_class_mask
        else:  # extract
            conf_mask = conf_mask & sem_class_mask

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    # Apply semantic coloring AFTER filtering if SAM is enabled
    if enable_sam and semantic_colors_full is not None:
        print("Applying semantic coloring to point cloud...")
        print(f"Conf mask shape: {conf_mask.shape}, True count: {conf_mask.sum()}")
        print(f"Semantic colors full shape: {semantic_colors_full.shape}")

        # Check if shapes match
        if semantic_colors_full.shape[0] == conf_mask.shape[0]:
            # Apply the same mask to semantic colors
            semantic_colors = semantic_colors_full[conf_mask]
            # Blend original colors with semantic colors
            colors_rgb = blend_semantic_colors(colors_rgb, semantic_colors, alpha=0.7)
        else:
            print(f"Warning: Cannot apply semantic colors - shape mismatch!")
            print(f"  semantic_colors_full: {semantic_colors_full.shape[0]}")
            print(f"  conf_mask: {conf_mask.shape[0]}")
            print("  Skipping semantic coloring for point cloud.")

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    if show_cam:
        # Add camera models to the scene
        for i in range(num_cameras):
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    # Align scene to the observation of the first camera
    scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)

    print("GLB Scene built")
    return scene_3d


def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(scene_3d: trimesh.Scene, extrinsics_matrices: np.ndarray) -> trimesh.Scene:
    """
    Aligns the 3D scene based on the extrinsics of the first camera.

    Args:
        scene_3d (trimesh.Scene): The 3D scene to be aligned.
        extrinsics_matrices (np.ndarray): Camera extrinsic matrices.

    Returns:
        trimesh.Scene: Aligned 3D scene.
    """
    # Set transformations for scene alignment
    opengl_conversion_matrix = get_opengl_conversion_matrix()

    # Rotation matrix for alignment (180 degrees around the y-axis)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    # Apply transformation
    initial_transformation = np.linalg.inv(extrinsics_matrices[0]) @ opengl_conversion_matrix @ align_rotation
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(-1, -2)  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image_path: Path to input image
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    assert mask_filename is not None
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def generate_depth_visualizations(predictions, target_dir):
    """
    Generate colormapped depth map images for each frame.
    Returns a sorted list of saved PNG file paths.
    """
    import zipfile

    depth = predictions["depth"]  # (S, H, W, 1)
    if depth.ndim == 4:
        depth = depth[..., 0]  # (S, H, W)

    S = depth.shape[0]
    out_dir = os.path.join(target_dir, "depth_vis")
    os.makedirs(out_dir, exist_ok=True)

    colormap = matplotlib.colormaps["turbo"]
    paths = []

    for s in range(S):
        d = depth[s]  # (H, W)
        valid = d > 1e-8
        if valid.any():
            d_min, d_max = d[valid].min(), d[valid].max()
            if d_max - d_min > 1e-8:
                d_norm = (d - d_min) / (d_max - d_min)
            else:
                d_norm = np.zeros_like(d)
        else:
            d_norm = np.zeros_like(d)
        d_norm = np.clip(d_norm, 0, 1)

        rgb = (colormap(d_norm)[:, :, :3] * 255).astype(np.uint8)
        # Mark invalid pixels as black
        rgb[~valid] = 0

        path = os.path.join(out_dir, f"depth_{s:04d}.png")
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        paths.append(path)

    # Create zip
    zip_path = os.path.join(target_dir, "depth_maps.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, os.path.basename(p))

    return sorted(paths)


def get_semantic_colors(semantic_masks, selected_frame_idx=None):
    """
    Generate semantic colors for each pixel based on semantic masks.

    Args:
        semantic_masks: Array of shape (S, H, W) with semantic IDs
        selected_frame_idx: Optional frame index to filter

    Returns:
        colors: Array of shape (N, 3) with RGB colors for each point
    """
    # Define color palette for different semantic IDs
    color_palette = {
        0: [128, 128, 128],  # Background - gray
        1: [255, 0, 0],      # Semantic 1 - red
        2: [0, 255, 0],      # Semantic 2 - green
        3: [0, 0, 255],      # Semantic 3 - blue
        4: [255, 255, 0],    # Semantic 4 - yellow
        5: [255, 0, 255],    # Semantic 5 - magenta
        6: [0, 255, 255],    # Semantic 6 - cyan
        7: [255, 128, 0],    # Semantic 7 - orange
        8: [128, 0, 255],    # Semantic 8 - purple
        9: [0, 255, 128],    # Semantic 9 - lime
    }

    if selected_frame_idx is not None:
        semantic_masks = semantic_masks[selected_frame_idx][None]

    # Flatten semantic masks
    semantic_ids = semantic_masks.reshape(-1)

    # Map semantic IDs to colors
    colors = np.zeros((len(semantic_ids), 3), dtype=np.uint8)
    for sem_id, color in color_palette.items():
        mask = semantic_ids == sem_id
        colors[mask] = color

    return colors


def blend_semantic_colors(original_colors, semantic_colors, alpha=0.7):
    """
    Blend original RGB colors with semantic colors.

    Args:
        original_colors: Array of shape (N, 3) with original RGB colors
        semantic_colors: Array of shape (N, 3) with semantic RGB colors
        alpha: Blending factor (0 = all original, 1 = all semantic)

    Returns:
        blended_colors: Array of shape (N, 3) with blended RGB colors
    """
    # Ensure both arrays have the same shape
    if original_colors.shape != semantic_colors.shape:
        print(f"Warning: Shape mismatch in blend_semantic_colors: {original_colors.shape} vs {semantic_colors.shape}")
        return original_colors

    # Only blend where semantic color is not gray (background)
    is_semantic = (semantic_colors != [128, 128, 128]).any(axis=1)

    blended = original_colors.copy()
    if is_semantic.any():
        blended[is_semantic] = (
            alpha * semantic_colors[is_semantic] +
            (1 - alpha) * original_colors[is_semantic]
        ).astype(np.uint8)

    return blended


def generate_semantic_depth_visualizations(predictions, target_dir, semantic_masks):
    """
    Generate semantic-colored depth map images for each frame.
    Returns a sorted list of saved PNG file paths.
    """
    import zipfile

    depth = predictions["depth"]  # (S, H, W, 1)
    if depth.ndim == 4:
        depth = depth[..., 0]  # (S, H, W)

    S = depth.shape[0]
    out_dir = os.path.join(target_dir, "semantic_depth_vis")
    os.makedirs(out_dir, exist_ok=True)

    colormap = matplotlib.colormaps["turbo"]
    paths = []

    for s in range(S):
        d = depth[s]  # (H, W)
        valid = d > 1e-8

        # Generate depth colormap
        if valid.any():
            d_min, d_max = d[valid].min(), d[valid].max()
            if d_max - d_min > 1e-8:
                d_norm = (d - d_min) / (d_max - d_min)
            else:
                d_norm = np.zeros_like(d)
        else:
            d_norm = np.zeros_like(d)
        d_norm = np.clip(d_norm, 0, 1)

        rgb = (colormap(d_norm)[:, :, :3] * 255).astype(np.uint8)

        # Apply semantic coloring
        if s < semantic_masks.shape[0]:
            sem_mask = semantic_masks[s]  # (H, W)

            # Generate semantic colors for this single frame
            # Get just this frame's mask and flatten it directly
            single_frame_mask = sem_mask.reshape(-1)  # Flatten to 1D

            # Map semantic IDs to colors
            color_palette = {
                0: [128, 128, 128],  # Background - gray
                1: [255, 0, 0],      # Semantic 1 - red
                2: [0, 255, 0],      # Semantic 2 - green
                3: [0, 0, 255],      # Semantic 3 - blue
                4: [255, 255, 0],    # Semantic 4 - yellow
                5: [255, 0, 255],    # Semantic 5 - magenta
                6: [0, 255, 255],    # Semantic 6 - cyan
                7: [255, 128, 0],    # Semantic 7 - orange
                8: [128, 0, 255],    # Semantic 8 - purple
                9: [0, 255, 128],    # Semantic 9 - lime
            }

            semantic_color_map = np.zeros((len(single_frame_mask), 3), dtype=np.uint8)
            for sem_id, color in color_palette.items():
                mask = single_frame_mask == sem_id
                semantic_color_map[mask] = color

            # Reshape to (H, W, 3)
            H, W = d.shape
            semantic_color_map = semantic_color_map.reshape(H, W, 3)

            # Blend depth colors with semantic colors
            is_semantic = sem_mask > 0
            if is_semantic.any():
                rgb[is_semantic] = (
                    0.5 * rgb[is_semantic] +
                    0.5 * semantic_color_map[is_semantic]
                ).astype(np.uint8)

        # Mark invalid pixels as black
        rgb[~valid] = 0

        path = os.path.join(out_dir, f"semantic_depth_{s:04d}.png")
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        paths.append(path)

    # Create zip
    zip_path = os.path.join(target_dir, "semantic_depth_maps.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, os.path.basename(p))

    return sorted(paths)


def generate_semantic_pointmap_visualizations(predictions, target_dir, semantic_masks):
    """
    Generate semantic-colored point map images for each frame.
    Returns a sorted list of saved PNG file paths.
    """
    import zipfile

    world_points = predictions["world_points"]  # (S, H, W, 3)
    S = world_points.shape[0]
    out_dir = os.path.join(target_dir, "semantic_pointmap_vis")
    os.makedirs(out_dir, exist_ok=True)

    # Global min/max for consistent coloring
    wp_flat = world_points.reshape(-1, 3)
    valid = np.isfinite(wp_flat).all(axis=1) & (np.abs(wp_flat).max(axis=1) < 1e6)
    if valid.any():
        g_min = wp_flat[valid].min(axis=0)
        g_max = wp_flat[valid].max(axis=0)
    else:
        g_min = np.zeros(3)
        g_max = np.ones(3)

    denom = g_max - g_min
    denom[denom < 1e-8] = 1.0

    paths = []
    for s in range(S):
        wp = world_points[s]  # (H, W, 3)
        norm = (wp - g_min) / denom
        norm = np.clip(norm, 0, 1)
        rgb = (norm * 255).astype(np.uint8)

        # Apply semantic coloring
        if s < semantic_masks.shape[0]:
            sem_mask = semantic_masks[s]  # (H, W)

            # Generate semantic colors for this single frame
            # Get just this frame's mask and flatten it directly
            single_frame_mask = sem_mask.reshape(-1)  # Flatten to 1D

            # Map semantic IDs to colors
            color_palette = {
                0: [128, 128, 128],  # Background - gray
                1: [255, 0, 0],      # Semantic 1 - red
                2: [0, 255, 0],      # Semantic 2 - green
                3: [0, 0, 255],      # Semantic 3 - blue
                4: [255, 255, 0],    # Semantic 4 - yellow
                5: [255, 0, 255],    # Semantic 5 - magenta
                6: [0, 255, 255],    # Semantic 6 - cyan
                7: [255, 128, 0],    # Semantic 7 - orange
                8: [128, 0, 255],    # Semantic 8 - purple
                9: [0, 255, 128],    # Semantic 9 - lime
            }

            semantic_color_map = np.zeros((len(single_frame_mask), 3), dtype=np.uint8)
            for sem_id, color in color_palette.items():
                mask = single_frame_mask == sem_id
                semantic_color_map[mask] = color

            # Reshape to (H, W, 3)
            H, W = wp.shape[:2]
            semantic_color_map = semantic_color_map.reshape(H, W, 3)

            # Blend position colors with semantic colors
            is_semantic = sem_mask > 0
            if is_semantic.any():
                rgb[is_semantic] = (
                    0.5 * rgb[is_semantic] +
                    0.5 * semantic_color_map[is_semantic]
                ).astype(np.uint8)

        path = os.path.join(out_dir, f"semantic_pointmap_{s:04d}.png")
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        paths.append(path)

    # Create zip
    zip_path = os.path.join(target_dir, "semantic_pointmap_maps.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, os.path.basename(p))

    return sorted(paths)


def generate_pointmap_visualizations(predictions, target_dir):
    """
    Generate position-colored point map images for each frame.
    XYZ world coordinates are mapped to RGB channels with global normalization.
    Returns a sorted list of saved PNG file paths.
    """
    import zipfile

    world_points = predictions["world_points"]  # (S, H, W, 3)
    S = world_points.shape[0]
    out_dir = os.path.join(target_dir, "pointmap_vis")
    os.makedirs(out_dir, exist_ok=True)

    # Global min/max across all frames for consistent coloring
    wp_flat = world_points.reshape(-1, 3)
    valid = np.isfinite(wp_flat).all(axis=1) & (np.abs(wp_flat).max(axis=1) < 1e6)
    if valid.any():
        g_min = wp_flat[valid].min(axis=0)  # (3,)
        g_max = wp_flat[valid].max(axis=0)  # (3,)
    else:
        g_min = np.zeros(3)
        g_max = np.ones(3)

    denom = g_max - g_min
    denom[denom < 1e-8] = 1.0

    paths = []
    for s in range(S):
        wp = world_points[s]  # (H, W, 3)
        norm = (wp - g_min) / denom
        norm = np.clip(norm, 0, 1)
        rgb = (norm * 255).astype(np.uint8)

        path = os.path.join(out_dir, f"pointmap_{s:04d}.png")
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        paths.append(path)

    # Create zip
    zip_path = os.path.join(target_dir, "pointmap_maps.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, os.path.basename(p))

    return sorted(paths)
