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
import pickle
from pathlib import Path
import shutil
from PIL import Image
from tqdm import tqdm
import json
from xvfbwrapper import Xvfb
import copy

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


def mixture_pdf(x, probs, means, scales):
    pdf = np.zeros_like(x, dtype=np.float64)
    for weight, mu, s in zip(probs, means, scales):
        pdf += weight * logistic.pdf(x, loc=mu, scale=s)
    return pdf


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration",
    )
    parser.add_argument(
        "run_directory",
        type=Path,
        help="run directory",
    )
    parser.add_argument(
        "path_to_pickled_3d_future_models", help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "scene_json",
        type=Path,
        help="Path to annotated info",
    )
    parser.add_argument("--debug", action="store_true")
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
    parser.add_argument(
        "--save_frames", help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--n_frames", type=int, default=360, help="Number of frames to be rendered"
    )
    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

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
        args.path_to_pickled_3d_future_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    # Create the scene and the behaviour list for simple-3dviz
    vdisplay = Xvfb(width=1280, height=740)
    vdisplay.start()

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

    with open(args.scene_json, "r") as f:
        subscene_info = json.load(f)

    # Get a floor plan
    vertices = np.array(subscene_info["vertices"])
    faces = np.array(subscene_info["faces"])

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

    room_mask = room_mask.to(device)
    boxes = {
        "class_labels": torch.zeros((1, 1, len(classes))).to(device),
        "translations": torch.zeros((1, 1, 3)).to(device),
        "sizes": torch.zeros((1, 1, 3)).to(device),
        "angles": torch.zeros((1, 1, 1)).to(device),
    }

    min_bound_translation, max_bound_translation = dataset.bounds["translations"]
    min_bound_size, max_bound_size = dataset.bounds["sizes"]
    min_bound_rotation, max_bound_rotation = dataset.bounds["angles"]
    for object_info in subscene_info["objects"]:
        our_translation = np.array(object_info["translation"])
        our_translation = rot_180_z.apply(our_translation)
        our_size = np.array(object_info["size"])
        our_translation[1] = our_size[1]

        normalized_translation = dataset.scale(
            our_translation, min_bound_translation, max_bound_translation
        )
        normalized_size = dataset.scale(our_size, min_bound_size, max_bound_size)
        our_rotation = np.array(object_info["rotation"])
        if our_rotation > max_bound_rotation:
            our_rotation -= 2 * np.pi

        normalized_rotation = dataset.scale(
            our_rotation, min_bound_rotation, max_bound_rotation
        )
        assert object_info["category"] in classes

        box = {
            "class_labels": torch.from_numpy(classes == object_info["category"])
            .float()
            .view(1, 1, len(classes))
            .to(device),
            "translations": torch.from_numpy(normalized_translation)
            .float()
            .view(1, 1, 3)
            .to(device),
            "sizes": torch.from_numpy(normalized_size)
            .float()
            .view(1, 1, 3)
            .to(device),
            "angles": torch.from_numpy(normalized_rotation)
            .float()
            .view(1, 1, 1)
            .to(device),
        }
        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

    # Extract the location params before end symbol
    query_category = subscene_info["query_object"]["category"]
    assert query_category in classes
    query_class_label = torch.from_numpy(classes == query_category)
    query_class_label = (
        query_class_label.float().view(1, 1, len(classes)).to(device)
    )

    weight_file = Path(args.run_directory) / "model_00175"
    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes, config, weight_file, device=device
    )
    network.eval()

    output_directory = Path(args.scene_json).parent / "atiss_mask"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True)
    save_dir = output_directory

    with torch.no_grad():
        dmll_params = network.distribution_translations(
            room_mask=room_mask,
            class_label=query_class_label,
            boxes=boxes,
            device=device,
        )

    x_dmll_params, _, z_dmll_params = dmll_params
    x_probs, x_means, x_scales = x_dmll_params
    x_probs, x_means, x_scales = (
        x_probs.squeeze().cpu().numpy(),
        x_means.squeeze().cpu().numpy(),
        x_scales.squeeze().cpu().numpy(),
    )
    z_probs, z_means, z_scales = z_dmll_params
    z_probs, z_means, z_scales = (
        z_probs.squeeze().cpu().numpy(),
        z_means.squeeze().cpu().numpy(),
        z_scales.squeeze().cpu().numpy(),
    )

    grid_size = 256
    cell_size = 6.2 / grid_size
    min_bound, max_bound = dataset.bounds["translations"]

    # global space
    x = np.linspace(3.1, -3.1, 256)
    z = np.linspace(-3.1, 3.1, 256)

    # convert to local space
    # mask values outside of range
    x_mask = np.logical_and(x > min_bound[0], x < max_bound[0])
    z_mask = np.logical_and(z > min_bound[2], z < max_bound[2])
    x = (x - min_bound[0]) / (max_bound[0] - min_bound[0])
    z = (z - min_bound[2]) / (max_bound[2] - min_bound[2])
    x = (2 * x) - 1
    z = (2 * z) - 1

    x_density = mixture_pdf(x, x_probs, x_means, x_scales)
    z_density = mixture_pdf(z, z_probs, z_means, z_scales)
    x_density[~x_mask] = 0
    z_density[~z_mask] = 0
    pdf = np.outer(x_density, z_density)

    location_pdf = pdf / pdf.max()

    if args.debug:
        # sample network and print results of those samples
        num_query_samples = 5
        with torch.no_grad():
            translations_pre = []
            for i in range(num_query_samples):
                box = network.autoregressive_decode_with_class_label(
                    room_mask=room_mask,
                    class_label=query_class_label,
                    boxes=boxes,
                )
                translations_pre.append(box["translations"].view(3).tolist())

                for k in box.keys():
                    boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        box = network.end_symbol(device)
        for k in box.keys():
            boxes[k] = torch.cat([boxes[k], box[k]], dim=1)

        bbox_params = {
            "class_labels": boxes["class_labels"].to("cpu"),
            "translations": boxes["translations"].to("cpu"),
            "sizes": boxes["sizes"].to("cpu"),
            "angles": boxes["angles"].to("cpu"),
        }

        boxes_post = dataset.post_process(bbox_params)
        translations = []
        angles = []

        for i in range(num_query_samples):
            i = i + 1 + len(subscene_info["objects"])
            translation = boxes_post["translations"][:, i:i+1, :].view(3).tolist()
            translation = rot_180_z.apply(translation).tolist()
            angle = boxes_post["angles"][:, i:i+1, :].view(1).tolist()
            translations.append(translation)
            angles.append(angle)

        bbox_params_t = (
            torch.cat(
                [
                    boxes_post["class_labels"],
                    boxes_post["translations"],
                    boxes_post["sizes"],
                    boxes_post["angles"],
                ],
                dim=-1,
            )
            .cpu()
            .numpy()
        )

        renderables, _, _ = get_textured_objects(
            bbox_params_t, objects_dataset, classes
        )
        query_renderables = renderables[-num_query_samples:]
        renderables = renderables[:-num_query_samples]

        renderables += floor_plan

        # do rendering for scene without objects
        path_to_image = save_dir / f"scene"
        behaviours = [
            LightToCamera(),
            SaveFrames(str(path_to_image) + ".png", 1),
        ]

        render(
            renderables,
            behaviours=behaviours,
            size=args.window_size,
            camera_position=args.camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=args.background,
            n_frames=args.n_frames,
            scene=scene,
        )

        mask = location_pdf > 0.5 
        path_to_image = path_to_image.with_suffix(".png")
        scene_image = np.array(Image.open(path_to_image))
        scene_image[mask] = [255, 0, 0, 255]
        Image.fromarray(scene_image).save(path_to_image)

        np.savez(save_dir / "masks.npz", masks = [mask])
        for query_idx, query_renderable in enumerate(query_renderables):
            path_to_image = save_dir / f"scene_with_query_{query_idx}"
            behaviours = [
                LightToCamera(),
                SaveFrames(str(path_to_image) + ".png", 1),
            ]

            render(
                renderables + [query_renderable],
                behaviours=behaviours,
                size=args.window_size,
                camera_position=args.camera_position,
                camera_target=args.camera_target,
                up_vector=args.up_vector,
                background=args.background,
                n_frames=args.n_frames,
                scene=scene,
            )

            sample_folder = save_dir / "samples" / f"sample_{query_idx}"
            if sample_folder.exists():
                shutil.rmtree(sample_folder)
            sample_folder.mkdir(parents=True)

            subscene_info_copy = copy.deepcopy(subscene_info)
            query_info = subscene_info_copy["query_object"]
            query_info["translation"] = translations[query_idx] 
            query_info["rotation"] = angles[query_idx]
            subscene_info_copy["objects"].append(query_info)
            with open(sample_folder / "scene.json", "w") as f:
                json.dump(subscene_info_copy, f)

    vdisplay.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
