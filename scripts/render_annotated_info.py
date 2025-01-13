#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

"""Script used for generating scenes using a previously trained model."""
import argparse
import logging
import json
import os
import sys
import pickle
from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
from pyrr import Matrix44
from scipy.stats import logistic
from scipy.spatial.transform import Rotation

from training_utils import load_config

from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects
from scene_synthesis.datasets.base import THREED_FRONT_BEDROOM_FURNITURE

from simple_3dviz import Mesh, Scene
from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render

from utils import render as utils_render, floor_plan_renderable


def snap_angle(angle):
    bin_width = (2 * np.pi) / 4
    angle = 2 * np.pi + angle if angle < 0 else angle
    angle_idx = int(round(angle / bin_width)) % 4
    return angle_idx * bin_width


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration",
    )
    parser.add_argument("output_directory", help="Path to the output directory")
    parser.add_argument(
        "path_to_pickled_3d_futute_models", help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "annotated_info_path",
        help="Path to annotated info",
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene",
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="1,0,0",
        help="Up vector of the scene",
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,4,0",
        help="Camer position in the scene",
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera",
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="256,256",
        help="Define the size of the scene and the window",
    )
    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Check if output directory exists and if it doesn't create it
    if os.path.exists(args.output_directory):
        shutil.rmtree(args.output_directory)
    os.makedirs(args.output_directory)

    config = load_config(args.config_file)

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"], split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["training"].get("splits", ["train", "val"]),
    )

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position
    room_side = 3.1
    scene.camera_matrix = Matrix44.orthogonal_projection(
        left=-room_side,
        right=room_side,
        bottom=room_side,
        top=-room_side,
        near=0.1,
        far=6,
    )

    classes = np.array(dataset.class_labels)

    annotated_info_path = Path(args.annotated_info_path)
    output_directory = Path(args.output_directory)
    subscene_info_jsons = []
    info_jsons = []
    output_paths = []
    for batch_path in annotated_info_path.iterdir():
        batch_name = batch_path.name
        for subscene_folder in batch_path.iterdir():
            with open(subscene_folder / "subscene_info.json", "r") as f:
                subscene_info = json.load(f)
            with open(subscene_folder / "info.json", "r") as f:
                info = json.load(f)
            subscene_info_jsons.append(subscene_info)
            info_jsons.append(info)

            output_path = output_directory / batch_name / subscene_folder.name
            output_paths.append(output_path)

    scene_id_to_room = {str(room.scene_id): room for room in raw_dataset}
    for subscene_info, info_json, output_path in tqdm(
        zip(subscene_info_jsons, info_jsons, output_paths),
        total=len(subscene_info_jsons),
    ):
        output_path.mkdir(parents=True)

        # Get a floor plan
        vertices = np.array(subscene_info["vertices"])
        faces = np.array(subscene_info["faces"])

        special_cases = []

        # library
        # special_cases = ['71098', '62499', '66674', '70803', '2931', '61060', '61592', '63804', '59720', '56384', '59439', '60574', '60568', '64100', '57957', '64770']

        # diningroom
        # special_cases = ['54593', '1422719', '1643180', '50185']
        if output_path.stem in special_cases:
            min_bound = np.amin(vertices, axis=0)
            max_bound = np.amax(vertices, axis=0)
            vertices = np.array(
                [
                    [min_bound[0], 0, min_bound[2]],
                    [min_bound[0], 0, max_bound[2]],
                    [max_bound[0], 0, max_bound[2]],
                    [max_bound[0], 0, min_bound[2]],
                ]
            )
            faces = np.array([[0, 1, 2], [0, 2, 3]])

        # Apply correction to align with our rendering
        rot_180_z = Rotation.from_rotvec([0, 0, np.pi])
        vertices = rot_180_z.apply(vertices)
        faces = faces[:, ::-1]

        floor_plan = Mesh.from_faces(vertices, faces, (0.5, 0.5, 0.5, 1.0))
        floor_plan = [floor_plan]

        empty_box = {
            "class_labels": torch.zeros((1, 1, len(classes))),
            "translations": torch.zeros((1, 1, 3)),
            "sizes": torch.zeros((1, 1, 3)),
            "angles": torch.zeros((1, 1, 1)),
        }
        boxes = dict(empty_box)
        object_info_list = subscene_info["objects"] + [subscene_info["query_object"]]
        for object_info in object_info_list:
            translation = np.array(object_info["translation"])
            translation = rot_180_z.apply(translation)
            size = np.array(object_info["size"])
            translation[1] = size[1]

            box = {
                "class_labels": torch.from_numpy(classes == object_info["category"])
                .float()
                .view(1, 1, len(classes)),
                "translations": torch.from_numpy(translation).float().view(1, 1, 3),
                "sizes": torch.from_numpy(size).float().view(1, 1, 3),
                "angles": torch.from_numpy(np.array(object_info["rotation"]))
                .float()
                .view(1, 1, 1),
            }
            for k in box.keys():
                boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        for k in empty_box.keys():
            boxes[k] = torch.cat([boxes[k], empty_box[k]], dim=1)

        bbox_params_t = (
            torch.cat(
                [
                    boxes["class_labels"],
                    boxes["translations"],
                    boxes["sizes"],
                    boxes["angles"],
                ],
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        renderables, _ = get_textured_objects(bbox_params_t, objects_dataset, classes)
        query_renderable = renderables[-1]
        original_scene_renderables = renderables + floor_plan
        scene_renderables = renderables[:-1] + floor_plan

        # Do the rendering
        path_to_image = output_path / f"scene"
        behaviours = [LightToCamera(), SaveFrames(str(path_to_image) + ".png", 1)]

        render(
            scene_renderables,
            behaviours=behaviours,
            size=args.window_size,
            camera_position=args.camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=args.background,
            n_frames=1,
            scene=scene,
        )

        # Do the rendering
        path_to_image = output_path / f"original_scene"
        behaviours = [LightToCamera(), SaveFrames(str(path_to_image) + ".png", 1)]

        render(
            original_scene_renderables,
            behaviours=behaviours,
            size=args.window_size,
            camera_position=args.camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=args.background,
            n_frames=1,
            scene=scene,
        )

        grid_size = args.window_size[0]
        assert grid_size == args.window_size[1]
        cell_size = (2 * room_side) / grid_size
        corner_pos = [-room_side, 0, -room_side]

        # Get the query object from the original image and just crop out the boox area
        original_scene_image = Image.open(path_to_image.with_suffix(".png"))
        original_scene_image = np.array(original_scene_image)

        min_bound, max_bound = query_renderable.bbox
        min_bound_grid = (min_bound - corner_pos) / cell_size
        max_bound_grid = (max_bound - corner_pos) / cell_size
        x_min_grid = 255 - int(max_bound_grid[0])
        x_max_grid = 255 - int(min_bound_grid[0])
        z_min_grid = int(min_bound_grid[2])
        z_max_grid = int(max_bound_grid[2])
        query_image = original_scene_image[x_min_grid:x_max_grid, z_min_grid:z_max_grid]
        query_image = Image.fromarray(query_image)
        query_rotation = snap_angle(subscene_info["query_object"]["rotation"][0])
        query_rotation = (180 / np.pi) * query_rotation
        query_rotation = 360 - query_rotation
        query_image = query_image.rotate(query_rotation, expand=True)
        query_image.save(output_path / "query_object.png")

        with open(output_path / "info.json", "w") as f:
            json.dump(info_json, f, indent=4)

        # Also create a bar at the top that labels what all the objects in the scene are
        scene_image = Image.open(output_path / "scene.png")
        scene_image = np.array(scene_image)

        top_bar_images = []
        for scene_object in renderables[:-1]:
            min_bound, max_bound = scene_object.bbox
            min_bound_grid = (min_bound - corner_pos) / cell_size
            max_bound_grid = (max_bound - corner_pos) / cell_size

            x_min_grid = 255 - int(max_bound_grid[0])
            x_max_grid = 255 - int(min_bound_grid[0])
            z_min_grid = int(min_bound_grid[2])
            z_max_grid = int(max_bound_grid[2])

            # Give the bounds some padding
            x_min_grid = max(0, x_min_grid - 25)
            x_max_grid = min(255, x_max_grid + 25)
            z_min_grid = max(0, z_min_grid - 25)
            z_max_grid = min(255, z_max_grid + 25)

            scene_object_image = scene_image[
                x_min_grid:x_max_grid, z_min_grid:z_max_grid
            ]
            scene_object_image = Image.fromarray(np.uint8(scene_object_image))
            top_bar_images.append(scene_object_image)

        num_scene_objects = len(renderables[:-1])
        if num_scene_objects != 0:
            fig, axes = plt.subplots(
                1, num_scene_objects, squeeze=True, figsize=(num_scene_objects * 3, 3)
            )
            for i in range(num_scene_objects):
                object_category = subscene_info["objects"][i]["category"]
                if num_scene_objects == 1:
                    ax = axes
                else:
                    ax = axes[i]

                ax.set_title(object_category)
                ax.imshow(top_bar_images[i])
                ax.set_xticks([])
                ax.set_yticks([])

            fig.savefig(output_path / "key.png")
            plt.close(fig)
        else:
            key_img = np.ones((256, 64, 3))
            Image.fromarray(np.uint8(key_img * 255)).save(output_path / "key.png")


if __name__ == "__main__":
    main(sys.argv[1:])
