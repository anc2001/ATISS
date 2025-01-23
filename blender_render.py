import bpy
import os
import sys
from pathlib import Path
import json
from mathutils import Matrix, Vector
import shutil

def clear_scene():
    """Clear the current scene of all objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_meshes_from_folder(folder_path):
    """Import all mesh files from the specified folder."""
    colors_path = Path(folder_path) / "colors.json"
    colors = None
    if colors_path.exists():
        with open(colors_path, 'r') as f:
            colors = json.load(f)

    for filepath in folder_path.iterdir():
        if filepath.suffix == ('.obj'):
            bpy.ops.wm.obj_import(filepath=str(filepath))

            if colors is not None:
                key = str(int(filepath.stem.split("object_")[1]))
                if key in colors:
                    color = colors[key]
                    # gamma correct color
                    color = [x**2.2 for x in color]

                    imported_object = bpy.context.selected_objects[0]

                    if imported_object.data.materials:
                        imported_object.data.materials.clear()

                    # Create a new material
                    mat = bpy.data.materials.new(name="SolidColorMaterial")
                    mat.use_nodes = True

                    # Access the material's node tree
                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links

                    # Clear existing nodes
                    for node in nodes:
                        nodes.remove(node)

                    # Add a Principled BSDF shader
                    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
                    bsdf_node.location = (0, 0)

                    # Set the base color
                    bsdf_node.inputs['Base Color'].default_value = color 

                    # Add a Material Output node
                    output_node = nodes.new(type='ShaderNodeOutputMaterial')
                    output_node.location = (200, 0)

                    # Connect the BSDF to the Material Output
                    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

                    # Assign the material to the object
                    imported_object.data.materials.append(mat)

def setup_lighting(query=False):
    """Add lighting to the scene."""
    # Clear existing lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    if query:
        # Add a sun lamp
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
        sun = bpy.context.object
        sun.data.energy = 3.0

        location = (0, -10, 0)
        bpy.ops.object.light_add(type='AREA')
        area = bpy.context.object
        area.data.energy = 500
        area.data.size = 5

        up_vector = Vector((0, 0, 1))
        look_vector = -(Vector((0, 0, 0)) - Vector(location)).normalized()
        z_axis = look_vector
        x_axis = up_vector.cross(z_axis).normalized()
        y_axis = z_axis.cross(x_axis).normalized()
        rotation_matrix = Matrix((x_axis, y_axis, z_axis)).transposed()
        area.matrix_world = Matrix.Translation(location) @ rotation_matrix.to_4x4()
    else:
#        # 3 point lighting
#        bpy.ops.object.light_add(type='POINT', location=(5, -5, 5))
#        light = bpy.context.object
#        light.data.energy = 1000
#
#        bpy.ops.object.light_add(type='POINT', location=(-5, -5, 5))
#        light = bpy.context.object
#        light.data.energy = 1000
#
#        bpy.ops.object.light_add(type='POINT', location=(-5, 5, 5))
#        light = bpy.context.object
#        light.data.energy = 500
#
#        bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
#        sun = bpy.context.object
#        sun.data.energy = 3.0
#
        location = (5, 5, 10)
        bpy.ops.object.light_add(type='AREA')
        area = bpy.context.object
        area.data.energy = 1000 
        area.data.size = 5

        up_vector = Vector((0, 0, 1))
        look_vector = -(Vector((0, 0, 0)) - Vector(location)).normalized()
        z_axis = look_vector
        x_axis = up_vector.cross(z_axis).normalized()
        y_axis = z_axis.cross(x_axis).normalized()
        rotation_matrix = Matrix((x_axis, y_axis, z_axis)).transposed()
        area.matrix_world = Matrix.Translation(location) @ rotation_matrix.to_4x4()


def set_camera(location, up_vector, target_position):
    """Add a camera at the specified location, aligned to look at the target position."""
    bpy.ops.object.camera_add()
    camera = bpy.context.object

    # Align the camera to the target position
    bpy.context.view_layer.objects.active = camera
    bpy.ops.object.mode_set(mode='OBJECT')

    # Compute the direction to look at
    up_vector = Vector(up_vector)
    look_vector = -(Vector(target_position) - Vector(location)).normalized()
    z_axis = look_vector
    x_axis = up_vector.cross(z_axis).normalized()
    y_axis = z_axis.cross(x_axis).normalized()
    rotation_matrix = Matrix((x_axis, y_axis, z_axis)).transposed()

    # Apply the rotation to the camera
    camera.matrix_world = Matrix.Translation(location) @ rotation_matrix.to_4x4()

    return camera

def render_viewpoints(output_folder, viewpoints, resolution=(1920, 1080)):
    """Render the scene from specified viewpoints."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set the render resolution
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]

    bpy.context.scene.render.engine = "CYCLES"

    # Set the device_type
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA" # or "OPENCL"
    bpy.context.scene.cycles.device = 'GPU'

    for i, (location, up_vector, target_position) in enumerate(viewpoints):
        # Set up the camera
        camera = set_camera(location, up_vector, target_position)
        bpy.context.scene.camera = camera

        # Render settings
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        output_path = os.path.join(output_folder, f'render_{i}.png')
        bpy.context.scene.render.filepath = output_path

        # Render the scene
        bpy.context.scene.render.film_transparent = True
        bpy.ops.render.render(write_still=True)

        # Remove the camera after rendering
        bpy.data.objects.remove(camera, do_unlink=True)

def process_folder(input_folder, resolution):
    if (input_folder / "scene_mesh").exists():
        viewpoints = [
            ((0, 0, 16), (1, 0, 0), (0, 0, 0)),  # (location, up_vector, target_position)
            # ((8, 8, 9.5), (0, 0, 1), (0, 0, 0)),
            # ((-8, 8, 9.5), (0, 0, 1), (0, 0, 0)),
            # ((-8, -8, 9.5), (0, 0, 1), (0, 0, 0)),
            # ((6, -6, 6), (0, 0, 1), (0, 0, 0)),
        ]

        clear_scene()
        import_meshes_from_folder(input_folder / "scene_mesh")
        setup_lighting()
        output_folder = input_folder / "blender_scene_renderings"
        if output_folder.exists():
            shutil.rmtree(output_folder)
        render_viewpoints(output_folder, viewpoints, resolution)

#    if (input_folder / "query_mesh").exists(): 
#        viewpoints = [
#            ((0, -7, 0), (0, 0, 1), (0, 0, 0))
#        ]
#
#        clear_scene()
#        import_meshes_from_folder(input_folder / "query_mesh")
#        setup_lighting(query=True)
#        render_viewpoints(input_folder / "blender_query_renderings", viewpoints, resolution)
#
#    if (input_folder / "samples").exists():
#        for samples_folder in (input_folder / "samples").iterdir():
#            viewpoints = [
#                # ((0, 0, 12), (1, 0, 0), (0, 0, 0)),  # (location, up_vector, target_position)
#                # ((8, 8, 10), (0, 0, 1), (0, 0, 0)),
#                # ((-8, 8, 10), (0, 0, 1), (0, 0, 0)),
#                # ((-8, -8, 10), (0, 0, 1), (0, 0, 0)),
#                # ((8, -8, 10), (0, 0, 1), (0, 0, 0)),
#                # ((0, -5, 3), (0, 0, 1), (0, 5, 0)),
#                ((6, -6, 6), (0, 0, 1), (0, 0, 0)),
#            ]
#
#            clear_scene()
#            import_meshes_from_folder(samples_folder / "scene_mesh")
#            setup_lighting()
#            render_viewpoints(samples_folder/ "blender_scene_renderings", viewpoints, resolution)
#
    print("Rendering complete!")


# Main script
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: blender --background --python script.py -- <input_folder> <resolution_x> <resolution_y>")
        sys.exit(1)

    # Parse command line arguments
    args = sys.argv[sys.argv.index("--") + 1:]
    resolution_x = int(args[1])
    resolution_y = int(args[2])
    resolution = (resolution_x, resolution_y)

    input_folder = args[0]
    input_folder = Path(input_folder)

    is_text_file = True 
    if is_text_file:
        with open(input_folder, "r") as f:
            paths = [Path(line.rstrip()) for line in f]
        for scene_folder_path in paths:
            process_folder(scene_folder_path, resolution)
    else:
        process_folder(input_folder, resolution)
