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
import trimesh
from utils import export_scene
import cv2

from xvfbwrapper import Xvfb

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
        description="Render scenes in a folder specified by json"
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration",
    )
    parser.add_argument(
        "path_to_pickled_3d_futute_models", help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "scene_folder",
        type=Path,
        help="Path to annotated info",
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene",
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

    config = load_config(args.config_file)

    _ , dataset = get_dataset_raw_and_encoded(
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

    classes = np.array(dataset.class_labels)

    vdisplay = Xvfb(width=1280, height=740)
    vdisplay.start()

    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size)
    scene.up_vector = (1, 0, 0)
    scene.camera_target = (0, 0, 0) 
    scene.camera_position = (0, 4, 0)
    scene.light = (0, 4, 0) 
    room_side = 3.1
    scene.camera_matrix = Matrix44.orthogonal_projection(
        left=-room_side,
        right=room_side,
        bottom=room_side,
        top=-room_side,
        near=0.1,
        far=6,
    )

    scene_folder = args.scene_folder
    with open(scene_folder / "scene.json", 'r') as f:
        subscene_info = json.load(f)

    # Get a floor plan
    vertices = np.array(subscene_info['vertices'])
    faces = np.array(subscene_info['faces'])

    # Apply correction to align with our rendering
    rot_180_z = Rotation.from_rotvec([0, 0, np.pi])
    vertices = rot_180_z.apply(vertices)
    faces = faces[:, ::-1]

    floor_plan = Mesh.from_faces(vertices, faces, (0.7, 0.7, 0.7, 1.0))
    floor_plan_tr = trimesh.Trimesh(vertices = vertices, faces = faces)
    floor_plan_tr.vertex_colors = (0.7, 0.7, 0.7, 1.0)
    floor_plan = [floor_plan]

    empty_box = {
        'class_labels' : torch.zeros((1, 1, len(classes))),
        'translations': torch.zeros((1, 1, 3)),
        'sizes' : torch.zeros((1, 1, 3)),
        'angles' : torch.zeros((1, 1, 1)),
    }
    boxes = dict()
    for k, v in empty_box.items():
        boxes[k] = torch.clone(v)

    object_info_list = subscene_info["objects"]

    corner_pos = np.array([-3.1, 0, -3.1])
    cell_size = 6.2 / 256
    for object_idx, object_info in enumerate(object_info_list): 
        if object_info["category"] in ["pendant_lamp", "ceiling_lamp"]:
            continue

        translation = np.array(object_info["translation"])
        translation = rot_180_z.apply(translation)
        size = np.array(object_info["size"])
        translation[1] = size[1]

        if object_idx == (len(object_info_list) - 1):
            # x goes (l -> r), (p -> n)
            placement_center = (translation - corner_pos) / cell_size
            placement_x, placement_y = int(placement_center[0]), int(placement_center[2])
            placement_x = 255 - placement_x
            bin_width = (2 * np.pi) / 4
            angle = object_info["rotation"][0]
            angle = 2 * np.pi + angle if angle < 0 else angle
            angle_idx = np.around(angle / bin_width).astype(int) % 4
            if angle_idx % 2 == 1:
                grid_width, grid_height = int(size[2] / cell_size), int(size[0] /cell_size)
            else:
                grid_width, grid_height = int(size[0] / cell_size), int(size[2] /cell_size)

            grid_width += 4
            grid_height += 4

            top_left = (placement_x - grid_width, placement_y - grid_height)
            bottom_right = (placement_x + grid_width, placement_y + grid_height)

        box = {
            "class_labels": torch.from_numpy(classes == object_info["category"])
            .float()
            .view(1, 1, len(classes)),
            "translations": torch.from_numpy(translation)
            .float()
            .view(1, 1, 3),
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

    renderables, tr_meshes, colors = get_textured_objects(
        bbox_params_t, 
        objects_dataset, 
        classes,
    )

    query_tr_mesh = tr_meshes[-1]
    query_renderable = renderables[-1]

    tr_meshes = tr_meshes[:-1]
    renderables = renderables[:-1]

    mask_folder = scene_folder / "masks"
    mask_folder.mkdir(exist_ok = True)

    # render scene query original query image 
    path_to_image = mask_folder / "scene_image.png"
    behaviors = [SaveFrames(str(path_to_image), 1)]

    renderables += floor_plan
    render(
        renderables + [query_renderable],
        behaviours=behaviors,
        size=args.window_size,
        camera_position=(0, 4, 0),
        camera_target=(0, 0, 0),
        up_vector=(1, 0, 0),
        background=args.background,
        n_frames=1,
        scene=scene,
    )
    # this is in BGR
    img = cv2.imread(path_to_image)
    img = cv2.rectangle(
        img, 
        (top_left[1], top_left[0]), 
        (bottom_right[1], bottom_right[0]), 
        (0, 255, 0), 
        2
    )
    cv2.imwrite(path_to_image, img)

    path_to_image = mask_folder / "scene_no_query.png"
    behaviors = [SaveFrames(str(path_to_image), 1)]

    render(
        renderables,
        behaviours=behaviors,
        size=args.window_size,
        camera_position=(0, 4, 0),
        camera_target=(0, 0, 0),
        up_vector=(1, 0, 0),
        background=args.background,
        n_frames=1,
        scene=scene,
    )

    scene_image = np.array(Image.open(path_to_image)) / 255.0
    # set opacity of scene image to be lower
    scene_image[..., 3]  = 0.5
    data = np.load(scene_folder  / "masks.npz")
    masks = data["masks"] 
    names = data["names"]
    assert len(names) == len(masks)
    for mask_idx in range(len(masks)):
        name = names[mask_idx]
        overlay = np.array(scene_image)

        mask = masks[mask_idx].astype(bool)

        mask_image = np.zeros((256, 256, 3))
        mask_image[mask] = [1.0, 0, 0]
        mask_image = Image.fromarray(np.uint8(mask_image * 255))

        overlay[mask] = [1.0, 0, 0, 1.0]
        overlay = Image.fromarray(np.uint8(overlay * 255))
        overlay.save(mask_folder / f"{name}.png")

    vdisplay.stop()

if __name__ == "__main__":
    main(sys.argv[1:])
