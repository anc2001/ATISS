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
    parser.add_argument(
        "--color_object_indices",
        action="store_true"
    )
    parser.add_argument(
        "--walls",
        action="store_true"
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
    json_paths = []
    json_paths.append(scene_folder / "scene.json")
    for sample_dir in (scene_folder / "samples").iterdir():
        json_paths.append(sample_dir / "scene.json")

    for json_path in json_paths:
        output_dir = json_path.parent
        with open(json_path, 'r') as f:
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

        contains_query = False
        if "query_object" in subscene_info:
            query_info = subscene_info["query_object"]
            object_info_list.append(query_info)
            contains_query = True


        for object_idx, object_info in enumerate(object_info_list): 
            if object_info["category"] in ["pendant_lamp", "ceiling_lamp"]:
                continue

            translation = np.array(object_info["translation"])
            translation = rot_180_z.apply(translation)
            size = np.array(object_info["size"])
            translation[1] = size[1]

            if contains_query and object_idx == (len(object_info_list) - 1):
                translation[1] = 0

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

        if args.color_object_indices:
            num_objects = bbox_params_t.shape[1] - 2
            if args.walls:
                num_objects += len(subscene_info["walls"])
            cmap = plt.get_cmap("tab20", num_objects) 
        else:
            cmap = None

        renderables, tr_meshes, colors = get_textured_objects(
            bbox_params_t, 
            objects_dataset, 
            classes,
            cmap=cmap,
            color_by_idx=args.color_object_indices
        )

        if args.walls:
            wall_renderables = []
            wall_tr_meshes = []
            wall_colors = []
            wall_height = 2
            for wall_idx, wall_info in enumerate(subscene_info["walls"]):
                wall_color = cmap(wall_idx + len(renderables))

                extent = np.array(wall_info["size"]) * 2
                extent[1] = wall_height 
                translation = wall_info["translation"]
                translation = rot_180_z.apply(translation)
                translation[1] = wall_height / 2
                theta = wall_info["rotation"]

                box_wall = trimesh.creation.box(extent)
                A = np.zeros((4, 4))
                A[0, 0] = np.cos(theta)
                A[0, 2] = -np.sin(theta)
                A[2, 0] = np.sin(theta)
                A[2, 2] = np.cos(theta)
                A[1, 1] = 1.
                A[:3, 3] = translation 
                A[3, 3] = 1
                box_wall.apply_transform(A)
                wall_tr_meshes.append(box_wall)

                wall_mesh_renderable = Mesh.from_faces(
                    box_wall.vertices, box_wall.faces, wall_color
                ) 
                wall_renderables.append(wall_mesh_renderable)

                wall_colors.append(wall_color)

        if contains_query:
            query_tr_mesh = tr_meshes[-1]
            renderables = renderables[:-1]
            tr_meshes = tr_meshes[:-1]

            query_folder = output_dir / "query_mesh"
            if query_folder.exists():
                shutil.rmtree(query_folder)
            query_folder.mkdir()

            export_scene(query_folder, [query_tr_mesh]) 
            if args.color_object_indices:
                color = colors[-1]
                colors = colors[:-1]
                with open(query_folder / "colors.json", "w") as f:
                    json.dump({0: color}, f, indent=4)

        mesh_folder = output_dir / "scene_mesh"
        if mesh_folder.exists():
            shutil.rmtree(mesh_folder)
        mesh_folder.mkdir()
        if args.walls:
            tr_meshes += wall_tr_meshes
            colors += wall_colors

        tr_meshes.append(floor_plan_tr)
        export_scene(mesh_folder, tr_meshes) 
        if args.color_object_indices:
            with open(mesh_folder / "colors.json", "w") as f:
                colors_dict = dict()
                for i in range(len(renderables)):
                    colors_dict[i] = colors[i]
                colors_dict = {i : colors[i] for i in range(len(colors))}
                json.dump(colors_dict, f, indent=4)

        # Do the rendering
        path_to_image = output_dir / "atiss_viz.png"
        behaviors = [SaveFrames(str(path_to_image), 1)]

        if args.walls:
            renderables += wall_renderables

        renderables += floor_plan
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

        if (output_dir / "masks.npz").exists():
            mask_folder = output_dir / "masks"
            mask_folder.mkdir(exist_ok = True)

            scene_image = np.array(Image.open(path_to_image))[..., :3] / 255.0
            masks = np.load(output_dir / "masks.npz")["masks"]
            for mask_idx, mask in enumerate(masks):
                if len(mask.shape) == 3:
                    mask_image_collapsed = np.sum(mask, axis=0).astype(bool).astype(float)
                    mask_image_collapsed = np.repeat(
                        np.expand_dims(mask_image_collapsed, axis=-1), 
                        3, 
                        axis=2
                    )
                    overlay_collapsed = np.clip(scene_image - mask_image_collapsed, 0, 1)
                    overlay_collapsed = Image.fromarray(np.uint8(overlay_collapsed * 255))
                    overlay_collapsed.save(mask_folder / f"mask_{mask_idx}_collapsed.png")

                    stacked_references = mask
                    top_row = np.array([])
                    bottom_row = np.array([])
                    for i in range(4):
                        mask_image = np.expand_dims(stacked_references[i], axis=2)
                        mask_image = np.repeat(mask_image, 3, axis=2)
                        image_slice = np.clip(scene_image - mask_image, 0, 1)

                        if i == 0:
                            top_row = image_slice
                        elif i == 1:
                            top_row = np.append(top_row, image_slice, axis=1)
                        elif i == 2:
                            bottom_row = image_slice
                        elif i == 3:
                            bottom_row = np.append(bottom_row, image_slice, axis=1)

                    image = np.append(top_row, bottom_row, axis=0)
                    full_image = Image.fromarray(np.uint8(image * 255))
                    full_image.save(mask_folder / f"mask_{mask_idx}.png")
                else:
                    mask_image_collapsed = np.repeat(
                        np.expand_dims(mask, axis=-1), 
                        3, 
                        axis=2
                    )
                    overlay_collapsed = np.clip(scene_image - mask_image_collapsed, 0, 1)
                    overlay_collapsed = Image.fromarray(np.uint8(overlay_collapsed * 255))
                    overlay_collapsed.save(mask_folder / f"mask_{mask_idx}_collapsed.png")

    vdisplay.stop()

if __name__ == "__main__":
    main(sys.argv[1:])
