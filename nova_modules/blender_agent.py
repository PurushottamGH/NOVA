"""
Nova Blender Agent
=====================
Generates and executes Blender Python (bpy) scripts from natural language.
Can create objects, apply effects, and run full scenes via Blender CLI.

Usage:
    agent = NovaBlenderAgent()
    script = agent.create_object("cube", location=(0, 0, 1), scale=(2, 2, 2))
    result = agent.execute_in_blender(script, blender_path="C:/Program Files/Blender/blender.exe")
"""

import re
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


class NovaBlenderAgent:
    """
    Blender automation agent for Nova.

    - Generate bpy scripts from natural language via NovaMind
    - Execute scripts in Blender's background mode
    - Pre-built templates for common objects and effects
    """

    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    #  AI-powered script generation
    # ------------------------------------------------------------------ #

    def generate_script(self, description: str, tokenizer, model) -> str:
        """
        Generate a Blender Python script from a natural language description
        using the NovaMind model.

        Args:
            description: What to create in Blender (e.g. "a red sphere on a plane").
            tokenizer: NovaMindTokenizer instance.
            model: NovaMind model instance.

        Returns:
            Generated Python/bpy script as a string.
        """
        prompt = (
            "Write a Blender Python bpy script that: " + description + "\n"
            "The script should:\n"
            "- Import bpy\n"
            "- Clear default objects\n"
            "- Create the requested scene\n"
            "- Set up a camera and lighting\n"
            "```python\nimport bpy\n"
        )

        # Encode and generate
        import torch

        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        model.eval()
        with torch.inference_mode():
            output_ids = model.generate(
                input_tensor,
                max_new_tokens=500,
                temperature=0.4,
                top_k=50,
                top_p=0.9,
            )

        generated = tokenizer.decode(output_ids[0].tolist())

        # Extract code block from response
        script = self._extract_code(generated)
        return script

    def _extract_code(self, text: str) -> str:
        """Extract Python code block from model output."""
        # Try to find code between ```python ... ```
        match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try ``` ... ```
        match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: find everything starting with "import bpy"
        match = re.search(r'(import bpy.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Last resort: return the full text
        return text.strip()

    # ------------------------------------------------------------------ #
    #  Blender execution
    # ------------------------------------------------------------------ #

    def execute_in_blender(
        self, script: str, blender_path: str = "blender"
    ) -> Dict:
        """
        Execute a bpy script in Blender's background mode.

        Args:
            script: Python/bpy script content to execute.
            blender_path: Path to the Blender executable.

        Returns:
            dict with keys: success, output, error
        """
        # Write script to temp file
        temp_path = Path(tempfile.gettempdir()) / "nova_blender_script.py"
        try:
            temp_path.write_text(script, encoding="utf-8")

            result = subprocess.run(
                [blender_path, "--background", "--python", str(temp_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
            }

        except FileNotFoundError:
            return {
                "success": False,
                "output": "",
                "error": (
                    f"Blender not found at '{blender_path}'. "
                    "Please provide the full path to the Blender executable."
                ),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Blender script timed out after {self.timeout}s",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Execution failed: {e}",
            }
        finally:
            if temp_path.exists():
                temp_path.unlink()

    # ------------------------------------------------------------------ #
    #  Pre-built object templates
    # ------------------------------------------------------------------ #

    def create_object(self, object_type: str, **params) -> str:
        """
        Return a bpy script that creates a common Blender object.

        Args:
            object_type: One of "cube", "sphere", "plane", "camera", "light", "material".
            **params: Object-specific parameters (location, scale, color, etc.).

        Returns:
            Complete bpy Python script as string.
        """
        object_type = object_type.lower().strip()

        location = params.get("location", (0, 0, 0))
        scale = params.get("scale", (1, 1, 1))
        name = params.get("name", object_type.capitalize())

        loc_str = f"({location[0]}, {location[1]}, {location[2]})"
        scale_str = f"({scale[0]}, {scale[1]}, {scale[2]})"

        if object_type == "cube":
            return self._template_cube(name, loc_str, scale_str)
        elif object_type == "sphere":
            segments = params.get("segments", 32)
            return self._template_sphere(name, loc_str, scale_str, segments)
        elif object_type == "plane":
            return self._template_plane(name, loc_str, scale_str)
        elif object_type == "camera":
            rotation = params.get("rotation", (1.1, 0, 0.8))
            rot_str = f"({rotation[0]}, {rotation[1]}, {rotation[2]})"
            return self._template_camera(name, loc_str, rot_str)
        elif object_type == "light":
            energy = params.get("energy", 1000)
            light_type = params.get("light_type", "POINT")
            return self._template_light(name, loc_str, energy, light_type)
        elif object_type == "material":
            color = params.get("color", (0.8, 0.1, 0.1, 1.0))
            target = params.get("target", None)
            return self._template_material(name, color, target)
        else:
            return f"# Unsupported object type: {object_type}\n# Supported: cube, sphere, plane, camera, light, material"

    def _template_cube(self, name: str, loc: str, scale: str) -> str:
        return (
            "import bpy\n\n"
            f"bpy.ops.mesh.primitive_cube_add(location={loc}, scale={scale})\n"
            f"bpy.context.active_object.name = '{name}'\n"
            f"print(f'Created cube: {name}')\n"
        )

    def _template_sphere(self, name: str, loc: str, scale: str, segments: int) -> str:
        return (
            "import bpy\n\n"
            f"bpy.ops.mesh.primitive_uv_sphere_add(segments={segments}, ring_count=16, location={loc}, scale={scale})\n"
            f"bpy.context.active_object.name = '{name}'\n"
            "bpy.ops.object.shade_smooth()\n"
            f"print(f'Created sphere: {name}')\n"
        )

    def _template_plane(self, name: str, loc: str, scale: str) -> str:
        return (
            "import bpy\n\n"
            f"bpy.ops.mesh.primitive_plane_add(location={loc}, scale={scale})\n"
            f"bpy.context.active_object.name = '{name}'\n"
            f"print(f'Created plane: {name}')\n"
        )

    def _template_camera(self, name: str, loc: str, rot: str) -> str:
        return (
            "import bpy\n"
            "import math\n\n"
            f"bpy.ops.object.camera_add(location={loc}, rotation={rot})\n"
            f"bpy.context.active_object.name = '{name}'\n"
            "bpy.context.scene.camera = bpy.context.active_object\n"
            f"print(f'Created camera: {name}')\n"
        )

    def _template_light(self, name: str, loc: str, energy: float, light_type: str) -> str:
        return (
            "import bpy\n\n"
            f"bpy.ops.object.light_add(type='{light_type}', location={loc})\n"
            f"bpy.context.active_object.name = '{name}'\n"
            f"bpy.context.active_object.data.energy = {energy}\n"
            f"print(f'Created {light_type} light: {name} (energy={energy})')\n"
        )

    def _template_material(self, name: str, color: tuple, target: Optional[str]) -> str:
        r, g, b = color[0], color[1], color[2]
        a = color[3] if len(color) > 3 else 1.0
        lines = [
            "import bpy\n",
            f"mat = bpy.data.materials.new(name='{name}')",
            "mat.use_nodes = True",
            "bsdf = mat.node_tree.nodes.get('Principled BSDF')",
            f"bsdf.inputs['Base Color'].default_value = ({r}, {g}, {b}, {a})",
        ]
        if target:
            lines.append(f"\nobj = bpy.data.objects.get('{target}')")
            lines.append("if obj and obj.data:")
            lines.append("    if not obj.data.materials:")
            lines.append("        obj.data.materials.append(mat)")
            lines.append("    else:")
            lines.append("        obj.data.materials[0] = mat")
            lines.append(f"    print(f'Applied material {name} to {target}')")
        else:
            lines.append(f"\nprint(f'Created material: {name} with color ({r}, {g}, {b})')")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    #  Visual effect snippets
    # ------------------------------------------------------------------ #

    def add_effect(self, effect_type: str) -> str:
        """
        Return a bpy script snippet for a visual effect.

        Args:
            effect_type: One of "glow", "motion_blur", "depth_of_field", "hdri_lighting".

        Returns:
            bpy Python script snippet as string.
        """
        effect_type = effect_type.lower().strip()

        if effect_type == "glow":
            return self._effect_glow()
        elif effect_type == "motion_blur":
            return self._effect_motion_blur()
        elif effect_type == "depth_of_field":
            return self._effect_depth_of_field()
        elif effect_type == "hdri_lighting":
            return self._effect_hdri_lighting()
        else:
            return f"# Unsupported effect: {effect_type}\n# Supported: glow, motion_blur, depth_of_field, hdri_lighting"

    def _effect_glow(self) -> str:
        return (
            "import bpy\n\n"
            "# Enable compositing for glare/glow\n"
            "bpy.context.scene.use_nodes = True\n"
            "tree = bpy.context.scene.node_tree\n"
            "nodes = tree.nodes\n"
            "links = tree.links\n\n"
            "# Clear existing nodes\n"
            "for node in nodes:\n"
            "    nodes.remove(node)\n\n"
            "# Create render layers and composite nodes\n"
            "render_layers = nodes.new(type='CompositorNodeRLayers')\n"
            "composite = nodes.new(type='CompositorNodeComposite')\n"
            "glare = nodes.new(type='CompositorNodeGlare')\n\n"
            "# Configure glare\n"
            "glare.glare_type = 'FOG_GLOW'\n"
            "glare.quality = 'HIGH'\n"
            "glare.threshold = 0.5\n"
            "glare.size = 6\n\n"
            "# Link: Render -> Glare -> Composite\n"
            "links.new(render_layers.outputs['Image'], glare.inputs['Image'])\n"
            "links.new(glare.outputs['Image'], composite.inputs['Image'])\n\n"
            "print('Added glow/glare effect')\n"
        )

    def _effect_motion_blur(self) -> str:
        return (
            "import bpy\n\n"
            "# Enable motion blur in render settings\n"
            "bpy.context.scene.render.use_motion_blur = True\n"
            "bpy.context.scene.render.motion_blur_shutter = 0.5\n"
            "bpy.context.scene.render.motion_blur_position = 'CENTER'\n\n"
            "# If using Cycles, set denoising\n"
            "if bpy.context.scene.render.engine == 'CYCLES':\n"
            "    bpy.context.scene.cycles.motion_blur_position = 'CENTER'\n\n"
            "print('Enabled motion blur (shutter=0.5)')\n"
        )

    def _effect_depth_of_field(self) -> str:
        return (
            "import bpy\n\n"
            "# Enable depth of field on the active camera\n"
            "cam = bpy.context.scene.camera\n"
            "if cam and cam.type == 'CAMERA':\n"
            "    cam.data.dof.use_dof = True\n"
            "    cam.data.dof.focus_distance = 5.0\n"
            "    cam.data.dof.aperture_fstop = 1.4\n"
            "    print(f'DOF enabled: focus={cam.data.dof.focus_distance}m, f/{cam.data.dof.aperture_fstop}')\n"
            "else:\n"
            "    print('No active camera found for DOF')\n"
        )

    def _effect_hdri_lighting(self) -> str:
        return (
            "import bpy\n"
            "import os\n\n"
            "# Set up HDRI environment lighting\n"
            "world = bpy.context.scene.world\n"
            "if not world:\n"
            "    world = bpy.data.worlds.new('NovaWorld')\n"
            "    bpy.context.scene.world = world\n\n"
            "world.use_nodes = True\n"
            "tree = world.node_tree\n"
            "nodes = tree.nodes\n"
            "links = tree.links\n\n"
            "# Clear existing world nodes\n"
            "for node in nodes:\n"
            "    nodes.remove(node)\n\n"
            "# Create environment texture setup\n"
            "bg = nodes.new(type='ShaderNodeBackground')\n"
            "bg.inputs['Strength'].default_value = 1.0\n\n"
            "env_tex = nodes.new(type='ShaderNodeTexEnvironment')\n"
            "# Set env_tex.image = bpy.data.images.load('path/to/your.hdr')\n\n"
            "tex_coord = nodes.new(type='ShaderNodeTexCoord')\n"
            "output = nodes.new(type='ShaderNodeOutputWorld')\n\n"
            "# Link nodes\n"
            "links.new(tex_coord.outputs['Generated'], env_tex.inputs['Vector'])\n"
            "links.new(env_tex.outputs['Color'], bg.inputs['Color'])\n"
            "links.new(bg.outputs['Background'], output.inputs['Surface'])\n\n"
            "print('HDRI lighting setup complete (load your .hdr file into env_tex.image)')\n"
        )
