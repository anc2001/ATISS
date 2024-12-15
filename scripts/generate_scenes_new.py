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
import os
import sys
import shutil
from pathlib import Path
from pyrr import Matrix44
from tqdm import tqdm
import numpy as np
import torch
import json
from PIL import Image
from scipy.spatial.transform import Rotation


from training_utils import load_config
from utils import floor_plan_from_scene, export_scene
from utils import render as utils_render

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects

from simple_3dviz import Mesh, Scene
from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="Path to the output directory"
    )
    parser.add_argument(
        "path_to_pickled_3d_future_models",
        help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="1,0,0",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,4,0",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="256,256",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=100,
        help="number of scenes to generate"
    )
    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    output_dir = args.output_directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    config = load_config(args.config_file)

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_future_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["val"])
        ),
        split=config["validation"].get("splits", ["val"])
    )
    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )

    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

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

    scene_indices = np.arange(len(dataset))
    np.random.shuffle(scene_indices)
    for scene_num in tqdm(range(args.num_scenes)):
        scene_idx = scene_indices[scene_num % len(dataset)]
        current_scene = raw_dataset[scene_idx]

        # Get a floor plan
        vertices, faces = current_scene.floor_plan
        vertices = vertices - current_scene.floor_plan_centroid

        # Apply correction to align with our rendering
        rot_180_z = Rotation.from_rotvec([0, 0, np.pi])
        vertices = rot_180_z.apply(vertices)
        faces = faces[:, ::-1]

        floor_plan = Mesh.from_faces(vertices, faces, (0.5, 0.5, 0.5, 1.0))
        floor_plan = [floor_plan]

        mask_floor_plan = Mesh.from_faces(vertices, faces, (1.0, 1.0, 1.0, 1.0))

        # Room mask rendered with (0, 0, -1) up camera position (0, 4, 0), room side 3.1
        mask_scene = Scene(size=(256, 256), background=(0, 0, 0, 1))
        mask_scene.up_vector = (0, 0, -1)
        mask_scene.camera_target = (0, 0, 0)
        mask_scene.camera_position = (0, 4, 0)
        mask_scene.light = (0, 4, 0)
        room_side = 3.1
        mask_scene.camera_matrix = Matrix44.orthogonal_projection(
            left=-room_side,
            right=room_side,
            bottom=room_side,
            top=-room_side,
            near=0.1,
            far=6,
        )

        room_mask = utils_render(
            mask_scene,
            [mask_floor_plan],
            (1.0, 1.0, 1.0),
            "flat",
        )
        room_mask = Image.fromarray(room_mask)
        room_mask = room_mask.resize(
            tuple(map(int, config["data"]["room_layout_size"].split(","))),
            resample=Image.BILINEAR,
        )
        room_mask = np.asarray(room_mask).astype(np.float32) / np.float32(255)

        room_mask = torch.from_numpy(
            np.transpose(room_mask[None, :, :, 0:1], (0, 3, 1, 2))
        ).float()

        bbox_params = network.generate_boxes(
            room_mask=room_mask.to(device),
            device=device
        )
        boxes = dataset.post_process(bbox_params)
        bbox_params_t = torch.cat([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], dim=-1).cpu().numpy()

        renderables, _ = get_textured_objects(
            bbox_params_t, objects_dataset, classes
        )
        renderables += floor_plan

        scene_output_dir = output_dir / f"scene_{scene_num:03d}"
        scene_output_dir.mkdir()

        # Export as json file
        output_dict = dict()
        output_dict["vertices"] = vertices.tolist()
        output_dict["faces"] = faces.tolist()
        output_dict["scene_id"] = str(current_scene.scene_id)

        objects = []
        num_objects = boxes["class_labels"].size(1)
        for box_idx in range(1, num_objects - 1):
            class_idx = boxes['class_labels'][0, box_idx, :].argmax().item()
            category = classes[class_idx]
            translation = boxes['translations'][0, box_idx, :].cpu().numpy()
            size = boxes['sizes'][0, box_idx, :].cpu().numpy()
            angle = - boxes['angles'][0, box_idx, :].cpu().numpy()

            object_info = {
                "category" : category, 
                "translation" : translation.tolist(),
                "size" : size.tolist(),
                "rotation" : angle.tolist()
            }
            objects.append(object_info)

        output_dict["objects"] = objects
        with open(scene_output_dir / "scene.json", "w") as f:
            json.dump(output_dict, f, indent=4)

        # Do the rendering
        path_to_image = scene_output_dir / f"rendering_viz"
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
