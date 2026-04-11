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

import ast
import re
import subprocess
import tempfile
from pathlib import Path


class NovaBlenderAgent:
    """
    Blender automation agent for Nova.

    - Generate bpy scripts from natural language via NovaMind
    - Execute scripts in Blender's background mode
    - Pre-built templates for common objects and effects
    - Validate and auto-repair scripts with retry logic
    - Build complete scenes from config dictionaries
    """

    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    #  AI-powered script generation
    # ------------------------------------------------------------------ #

    def generate_script(self, description: str, tokenizer, model) -> str:
        """
        Generate a Blender Python script from a natural language description.

        Args:
            description: What to create in Blender.
            tokenizer: NovaMindTokenizer instance.
            model: NovaMind model instance.

        Returns:
            Generated Python/bpy script as a string.
        """
        prompt = (
            "Write a Blender Python bpy script that: " + description + "\n"
            "The script should:\n- Import bpy\n- Clear default objects\n"
            "- Create the requested scene\n- Set up a camera and lighting\n"
            "```python\nimport bpy\n"
        )
        import torch

        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        model.eval()
        with torch.inference_mode():
            output_ids = model.generate(
                input_tensor, max_new_tokens=500, temperature=0.4, top_k=50, top_p=0.9
            )
        generated = tokenizer.decode(output_ids[0].tolist())
        script = self._extract_code(generated)
        return script

    def _extract_code(self, text: str) -> str:
        """Extract Python code block from model output."""
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"(import bpy.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    # ------------------------------------------------------------------ #
    #  Script validation
    # ------------------------------------------------------------------ #

    def validate_script(self, script: str) -> tuple[bool, str]:
        """
        Validate a Blender Python script for syntax and basic correctness.

        Args:
            script: Python/bpy script content to validate.

        Returns:
            Tuple of (is_valid, message). (True, "Valid") or (False, reason).
        """
        try:
            ast.parse(script)
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        if "import bpy" not in script:
            return False, "Script is missing 'import bpy' statement"
        return True, "Valid"

    # ------------------------------------------------------------------ #
    #  Blender execution
    # ------------------------------------------------------------------ #

    def execute_in_blender(self, script: str, blender_path: str = "blender") -> dict:
        """
        Execute a bpy script in Blender's background mode.

        Args:
            script: Python/bpy script content to execute.
            blender_path: Path to the Blender executable.

        Returns:
            dict with keys: success, output, error
        """
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
                "error": f"Blender not found at '{blender_path}'.",
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"Blender script timed out after {self.timeout}s",
            }
        except Exception as e:
            return {"success": False, "output": "", "error": f"Execution failed: {e}"}
        finally:
            if temp_path.exists():
                temp_path.unlink()

    # ------------------------------------------------------------------ #
    #  Execute with retry
    # ------------------------------------------------------------------ #

    def execute_with_retry(
        self, script: str, blender_path: str, model=None, tokenizer=None, max_retries: int = 3
    ) -> dict:
        """
        Execute a script with validation and retry logic.

        Args:
            script: Python/bpy script to execute.
            blender_path: Path to Blender executable.
            model: Optional NovaMind model for script repair.
            tokenizer: Optional tokenizer for script repair.
            max_retries: Maximum retry attempts (default 3).

        Returns:
            dict with keys: success, output, error, attempts
        """
        try:
            current_script = script
            for attempt in range(1, max_retries + 1):
                is_valid, reason = self.validate_script(current_script)
                if not is_valid:
                    if model is not None and tokenizer is not None:
                        repair_prompt = (
                            f"Fix this Blender script. Error: {reason}\n"
                            f"Original script:\n{current_script}"
                        )
                        current_script = self.generate_script(repair_prompt, tokenizer, model)
                        continue
                    return {
                        "success": False,
                        "output": "",
                        "error": f"Script invalid: {reason}. No model for repair.",
                        "attempts": attempt,
                    }
                result = self.execute_in_blender(current_script, blender_path)
                if result["success"]:
                    result["attempts"] = attempt
                    return result
                if model is not None and tokenizer is not None and attempt < max_retries:
                    repair_prompt = (
                        f"Fix this Blender script. Error: {result['error']}\n"
                        f"Original script:\n{current_script}"
                    )
                    current_script = self.generate_script(repair_prompt, tokenizer, model)
                else:
                    result["attempts"] = attempt
                    return result
            return {
                "success": False,
                "output": "",
                "error": f"Failed after {max_retries} attempts",
                "attempts": max_retries,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Execute with retry failed: {e}",
                "attempts": 0,
            }

    # ------------------------------------------------------------------ #
    #  Scene builder
    # ------------------------------------------------------------------ #

    def build_scene(self, config: dict) -> str:
        """
        Generate a complete bpy script from a scene configuration.

        Args:
            config: dict with keys: objects, lights, camera, render.

        Returns:
            Complete runnable bpy Python script as string.
        """
        try:
            lines = [
                "import bpy",
                "import math",
                "",
                "# === Clear Scene ===",
                "bpy.ops.object.select_all(action='SELECT')",
                "bpy.ops.object.delete(use_global=False)",
                "",
            ]
            for obj in config.get("objects", []):
                obj_type = obj.get("type", "cube")
                loc = obj.get("location", (0, 0, 0))
                name = obj.get("name", obj_type.capitalize())
                scale = obj.get("scale", (1, 1, 1))
                color = obj.get("color", None)
                obj_script = self.create_object(obj_type, location=loc, name=name, scale=scale)
                for line in obj_script.strip().split("\n"):
                    if not line.startswith("import bpy") and not line.startswith("import math"):
                        lines.append(line)
                if color:
                    ct = tuple(color[:3])
                    a = color[3] if len(color) > 3 else 1.0
                    mat_script = self._template_material(f"Mat_{name}", (*ct, a), name)
                    for line in mat_script.strip().split("\n"):
                        if not line.startswith("import bpy"):
                            lines.append(line)
                lines.append("")
            for i, light in enumerate(config.get("lights", [])):
                lt = light.get("type", "POINT").upper()
                loc = light.get("location", (0, 0, 5))
                energy = light.get("energy", 1000)
                lines.append(f"bpy.ops.object.light_add(type='{lt}', location={tuple(loc)})")
                lines.append(f"bpy.context.active_object.name = 'Light_{i}'")
                lines.append(f"bpy.context.active_object.data.energy = {energy}")
                lines.append("")
            cam = config.get("camera")
            if cam:
                cl = cam.get("location", (7, -6, 5))
                cr = cam.get("rotation", (1.1, 0, 0.8))
                lines.append(
                    f"bpy.ops.object.camera_add(location={tuple(cl)}, rotation={tuple(cr)})"
                )
                lines.append("bpy.context.active_object.name = 'SceneCamera'")
                lines.append("bpy.context.scene.camera = bpy.context.active_object")
                lines.append("")
            render = config.get("render")
            if render:
                engine = render.get("engine", "CYCLES").upper()
                if engine == "EEVEE":
                    engine = "BLENDER_EEVEE"
                lines.append(f"bpy.context.scene.render.engine = '{engine}'")
                samples = render.get("samples", 128)
                if "CYCLES" in engine:
                    lines.append(f"bpy.context.scene.cycles.samples = {samples}")
                else:
                    lines.append(f"bpy.context.scene.eevee.taa_render_samples = {samples}")
                output_path = render.get("output")
                if output_path:
                    lines.append(f"bpy.context.scene.render.filepath = '{output_path}'")
                    lines.append("bpy.ops.render.render(write_still=True)")
                lines.append("")
            lines.append("print('Scene build complete')")
            return "\n".join(lines) + "\n"
        except Exception as e:
            return f"# Error building scene: {e}\n"

    # ------------------------------------------------------------------ #
    #  Scene export
    # ------------------------------------------------------------------ #

    def export_scene(self, output_path: str, format: str = "OBJ") -> str:
        """
        Generate a bpy script that exports the scene.

        Args:
            output_path: File path for export.
            format: "OBJ" | "FBX" | "GLB" | "STL".

        Returns:
            bpy export script as string.
        """
        try:
            fmt = format.upper().strip()
            lines = ["import bpy", "", "# === Export Scene ==="]
            if fmt == "OBJ":
                lines.append(f"bpy.ops.wm.obj_export(filepath='{output_path}')")
            elif fmt == "FBX":
                lines.append(f"bpy.ops.export_scene.fbx(filepath='{output_path}')")
            elif fmt in ("GLB", "GLTF"):
                lines.append(f"bpy.ops.export_scene.gltf(filepath='{output_path}')")
            elif fmt == "STL":
                lines.append(f"bpy.ops.export_mesh.stl(filepath='{output_path}')")
            else:
                return f"# Unsupported format: {fmt}. Supported: OBJ, FBX, GLB, STL\n"
            lines.append(f"print('Exported to: {output_path} ({fmt})')")
            return "\n".join(lines) + "\n"
        except Exception as e:
            return f"# Export failed: {e}\n"

    # ------------------------------------------------------------------ #
    #  Pre-built object templates
    # ------------------------------------------------------------------ #

    def create_object(self, object_type: str, **params) -> str:
        """
        Return a bpy script that creates a Blender object.

        Args:
            object_type: cube/sphere/plane/camera/light/material/cylinder/cone/torus/text/empty.
            **params: location, scale, color, name, text, etc.

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
            return self._template_sphere(name, loc_str, scale_str, params.get("segments", 32))
        elif object_type == "plane":
            return self._template_plane(name, loc_str, scale_str)
        elif object_type == "camera":
            rot = params.get("rotation", (1.1, 0, 0.8))
            return self._template_camera(name, loc_str, f"({rot[0]}, {rot[1]}, {rot[2]})")
        elif object_type == "light":
            return self._template_light(
                name, loc_str, params.get("energy", 1000), params.get("light_type", "POINT")
            )
        elif object_type == "material":
            return self._template_material(
                name, params.get("color", (0.8, 0.1, 0.1, 1.0)), params.get("target")
            )
        elif object_type == "cylinder":
            return self._template_cylinder(name, loc_str, scale_str)
        elif object_type == "cone":
            return self._template_cone(name, loc_str, scale_str)
        elif object_type == "torus":
            return self._template_torus(name, loc_str, scale_str)
        elif object_type == "text":
            return self._template_text(name, loc_str, scale_str, params.get("text", "Nova"))
        elif object_type == "empty":
            return self._template_empty(name, loc_str, scale_str)
        else:
            return (
                f"# Unsupported object type: {object_type}\n"
                f"# Supported: cube, sphere, plane, camera, light, material, "
                f"cylinder, cone, torus, text, empty\n"
            )

    def _template_cube(self, name, loc, scale):
        return (
            f"import bpy\n\nbpy.ops.mesh.primitive_cube_add(location={loc}, scale={scale})\n"
            f"bpy.context.active_object.name = '{name}'\nprint(f'Created cube: {name}')\n"
        )

    def _template_sphere(self, name, loc, scale, segments):
        return (
            f"import bpy\n\nbpy.ops.mesh.primitive_uv_sphere_add(segments={segments}, "
            f"ring_count=16, location={loc}, scale={scale})\n"
            f"bpy.context.active_object.name = '{name}'\nbpy.ops.object.shade_smooth()\n"
            f"print(f'Created sphere: {name}')\n"
        )

    def _template_plane(self, name, loc, scale):
        return (
            f"import bpy\n\nbpy.ops.mesh.primitive_plane_add(location={loc}, scale={scale})\n"
            f"bpy.context.active_object.name = '{name}'\nprint(f'Created plane: {name}')\n"
        )

    def _template_camera(self, name, loc, rot):
        return (
            f"import bpy\nimport math\n\nbpy.ops.object.camera_add(location={loc}, rotation={rot})\n"
            f"bpy.context.active_object.name = '{name}'\n"
            f"bpy.context.scene.camera = bpy.context.active_object\n"
            f"print(f'Created camera: {name}')\n"
        )

    def _template_light(self, name, loc, energy, light_type):
        return (
            f"import bpy\n\nbpy.ops.object.light_add(type='{light_type}', location={loc})\n"
            f"bpy.context.active_object.name = '{name}'\n"
            f"bpy.context.active_object.data.energy = {energy}\n"
            f"print(f'Created {light_type} light: {name} (energy={energy})')\n"
        )

    def _template_material(self, name, color, target=None):
        """Generate bpy script to create and optionally apply a material.

        Args:
            name: Material name.
            color: RGBA tuple.
            target: Optional object name to apply material to.

        Returns:
            bpy script string.
        """
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
            lines += [
                f"\nobj = bpy.data.objects.get('{target}')",
                "if obj and obj.data:",
                "    if not obj.data.materials:",
                "        obj.data.materials.append(mat)",
                "    else:",
                "        obj.data.materials[0] = mat",
                f"    print(f'Applied material {name} to {target}')",
            ]
        else:
            lines.append(f"\nprint(f'Created material: {name} with color ({r}, {g}, {b})')")
        return "\n".join(lines) + "\n"

    def _template_cylinder(self, name, loc, scale):
        """Generate bpy script to create a cylinder."""
        return (
            f"import bpy\n\nbpy.ops.mesh.primitive_cylinder_add(location={loc}, scale={scale})\n"
            f"bpy.context.active_object.name = '{name}'\nprint(f'Created cylinder: {name}')\n"
        )

    def _template_cone(self, name, loc, scale):
        """Generate bpy script to create a cone."""
        return (
            f"import bpy\n\nbpy.ops.mesh.primitive_cone_add(location={loc}, scale={scale})\n"
            f"bpy.context.active_object.name = '{name}'\nprint(f'Created cone: {name}')\n"
        )

    def _template_torus(self, name, loc, scale):
        """Generate bpy script to create a torus."""
        return (
            f"import bpy\n\nbpy.ops.mesh.primitive_torus_add(location={loc})\n"
            f"bpy.context.active_object.scale = {scale}\n"
            f"bpy.context.active_object.name = '{name}'\nprint(f'Created torus: {name}')\n"
        )

    def _template_text(self, name, loc, scale, text):
        """Generate bpy script to create a text object."""
        return (
            f"import bpy\n\nbpy.ops.object.text_add(location={loc})\n"
            f"bpy.context.active_object.name = '{name}'\n"
            f"bpy.context.object.data.body = '{text}'\n"
            f"bpy.context.active_object.scale = {scale}\n"
            f"print(f'Created text: {name}')\n"
        )

    def _template_empty(self, name, loc, scale):
        """Generate bpy script to create an empty object."""
        return (
            f"import bpy\n\nbpy.ops.object.empty_add(location={loc})\n"
            f"bpy.context.active_object.name = '{name}'\n"
            f"bpy.context.active_object.scale = {scale}\n"
            f"print(f'Created empty: {name}')\n"
        )

    # ------------------------------------------------------------------ #
    #  Visual effect snippets
    # ------------------------------------------------------------------ #

    def add_effect(self, effect_type: str) -> str:
        """
        Return a bpy script snippet for a visual effect.

        Args:
            effect_type: glow/motion_blur/depth_of_field/hdri_lighting/
                         ambient_occlusion/bloom/wireframe/subdivision/bevel.

        Returns:
            bpy Python script snippet as string.
        """
        effect_type = effect_type.lower().strip()
        effects = {
            "glow": self._effect_glow,
            "motion_blur": self._effect_motion_blur,
            "depth_of_field": self._effect_depth_of_field,
            "hdri_lighting": self._effect_hdri_lighting,
            "ambient_occlusion": self._effect_ambient_occlusion,
            "bloom": self._effect_bloom,
            "wireframe": self._effect_wireframe,
            "subdivision": self._effect_subdivision,
            "bevel": self._effect_bevel,
        }
        if effect_type in effects:
            return effects[effect_type]()
        return f"# Unsupported effect: {effect_type}\n# Supported: {', '.join(effects.keys())}\n"

    def _effect_glow(self):
        return (
            "import bpy\n\nbpy.context.scene.use_nodes = True\n"
            "tree = bpy.context.scene.node_tree\nnodes = tree.nodes\nlinks = tree.links\n"
            "for node in nodes:\n    nodes.remove(node)\n"
            "render_layers = nodes.new(type='CompositorNodeRLayers')\n"
            "composite = nodes.new(type='CompositorNodeComposite')\n"
            "glare = nodes.new(type='CompositorNodeGlare')\n"
            "glare.glare_type = 'FOG_GLOW'\nglare.quality = 'HIGH'\n"
            "glare.threshold = 0.5\nglare.size = 6\n"
            "links.new(render_layers.outputs['Image'], glare.inputs['Image'])\n"
            "links.new(glare.outputs['Image'], composite.inputs['Image'])\n"
            "print('Added glow/glare effect')\n"
        )

    def _effect_motion_blur(self):
        return (
            "import bpy\n\nbpy.context.scene.render.use_motion_blur = True\n"
            "bpy.context.scene.render.motion_blur_shutter = 0.5\n"
            "bpy.context.scene.render.motion_blur_position = 'CENTER'\n"
            "if bpy.context.scene.render.engine == 'CYCLES':\n"
            "    bpy.context.scene.cycles.motion_blur_position = 'CENTER'\n"
            "print('Enabled motion blur (shutter=0.5)')\n"
        )

    def _effect_depth_of_field(self):
        return (
            "import bpy\n\ncam = bpy.context.scene.camera\n"
            "if cam and cam.type == 'CAMERA':\n"
            "    cam.data.dof.use_dof = True\n"
            "    cam.data.dof.focus_distance = 5.0\n"
            "    cam.data.dof.aperture_fstop = 1.4\n"
            "    print(f'DOF enabled: focus={cam.data.dof.focus_distance}m')\n"
            "else:\n    print('No active camera found for DOF')\n"
        )

    def _effect_hdri_lighting(self):
        return (
            "import bpy\nimport os\n\nworld = bpy.context.scene.world\n"
            "if not world:\n    world = bpy.data.worlds.new('NovaWorld')\n"
            "    bpy.context.scene.world = world\nworld.use_nodes = True\n"
            "tree = world.node_tree\nnodes = tree.nodes\nlinks = tree.links\n"
            "for node in nodes:\n    nodes.remove(node)\n"
            "bg = nodes.new(type='ShaderNodeBackground')\n"
            "bg.inputs['Strength'].default_value = 1.0\n"
            "env_tex = nodes.new(type='ShaderNodeTexEnvironment')\n"
            "tex_coord = nodes.new(type='ShaderNodeTexCoord')\n"
            "output = nodes.new(type='ShaderNodeOutputWorld')\n"
            "links.new(tex_coord.outputs['Generated'], env_tex.inputs['Vector'])\n"
            "links.new(env_tex.outputs['Color'], bg.inputs['Color'])\n"
            "links.new(bg.outputs['Background'], output.inputs['Surface'])\n"
            "print('HDRI lighting setup complete')\n"
        )

    def _effect_ambient_occlusion(self):
        """Generate bpy script to enable ambient occlusion."""
        return (
            "import bpy\n\n"
            "bpy.context.scene.world.light_settings.use_ambient_occlusion = True\n"
            "bpy.context.scene.world.light_settings.ao_factor = 1.0\n"
            "if bpy.context.scene.render.engine == 'BLENDER_EEVEE':\n"
            "    bpy.context.scene.eevee.use_gtao = True\n"
            "    bpy.context.scene.eevee.gtao_distance = 1.0\n"
            "print('Enabled ambient occlusion')\n"
        )

    def _effect_bloom(self):
        """Generate bpy script to enable bloom in EEVEE."""
        return (
            "import bpy\n\n"
            "if bpy.context.scene.render.engine == 'BLENDER_EEVEE':\n"
            "    bpy.context.scene.eevee.use_bloom = True\n"
            "    bpy.context.scene.eevee.bloom_threshold = 0.8\n"
            "    bpy.context.scene.eevee.bloom_intensity = 0.05\n"
            "    bpy.context.scene.eevee.bloom_radius = 6.5\n"
            "    print('Enabled bloom effect (EEVEE)')\n"
            "else:\n    print('Bloom is only available in EEVEE')\n"
        )

    def _effect_wireframe(self):
        """Generate bpy script to add wireframe modifier to active object."""
        return (
            "import bpy\n\nobj = bpy.context.active_object\n"
            "if obj and obj.type == 'MESH':\n"
            "    mod = obj.modifiers.new(name='Wireframe', type='WIREFRAME')\n"
            "    mod.thickness = 0.02\n    mod.use_replace = False\n"
            "    print(f'Added wireframe to {obj.name}')\n"
            "else:\n    print('No active mesh for wireframe')\n"
        )

    def _effect_subdivision(self):
        """Generate bpy script to add subdivision surface modifier."""
        return (
            "import bpy\n\nobj = bpy.context.active_object\n"
            "if obj and obj.type == 'MESH':\n"
            "    mod = obj.modifiers.new(name='Subdivision', type='SUBSURF')\n"
            "    mod.levels = 2\n    mod.render_levels = 2\n"
            "    print(f'Added subdivision (levels=2) to {obj.name}')\n"
            "else:\n    print('No active mesh for subdivision')\n"
        )

    def _effect_bevel(self):
        """Generate bpy script to add bevel modifier to active object."""
        return (
            "import bpy\n\nobj = bpy.context.active_object\n"
            "if obj and obj.type == 'MESH':\n"
            "    mod = obj.modifiers.new(name='Bevel', type='BEVEL')\n"
            "    mod.width = 0.1\n    mod.segments = 3\n"
            "    print(f'Added bevel (w=0.1, seg=3) to {obj.name}')\n"
            "else:\n    print('No active mesh for bevel')\n"
        )
