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

from utils import render as utils_render


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
        "subscene_info_path",
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

    _, dataset = get_dataset_raw_and_encoded(
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

    save_dir = Path(args.output_directory)
    with open(args.subscene_info_path, "r") as f:
        subscene_info = json.load(f)

    # Get a floor plan
    vertices = np.array(subscene_info["vertices"])
    faces = np.array(subscene_info["faces"])

    # Apply correction to align with our rendering
    rot_180_z = Rotation.from_rotvec([0, 0, np.pi])
    vertices = rot_180_z.apply(vertices)
    faces = faces[:, ::-1]

    floor_plan = Mesh.from_faces(vertices, faces, (0.7, 0.7, 0.7, 1.0))
    floor_plan = [floor_plan]

    empty_box = {
        "class_labels": torch.zeros((1, 1, len(classes))),
        "translations": torch.zeros((1, 1, 3)),
        "sizes": torch.zeros((1, 1, 3)),
        "angles": torch.zeros((1, 1, 1)),
    }
    boxes = dict()
    for k, v in empty_box.items():
        boxes[k] = torch.clone(v)
    for object_info in subscene_info["objects"]:
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
        boxes[k] = torch.cat([boxes[k], torch.clone(empty_box[k])], dim=1)

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
    renderables += floor_plan

    # Do the rendering
    path_to_image = save_dir / f"scene"
    behaviours = [LightToCamera(), SaveFrames(str(path_to_image) + ".png", 1)]

    render(
        renderables,
        behaviours=behaviours,
        size=args.window_size,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        up_vector=args.up_vector,
        background=args.background,
        n_frames=1,
        scene=scene,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
