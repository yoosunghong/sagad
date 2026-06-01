"""Build the M_SagadWPO master material in-editor (Phase 5, ARCHITECT sec. 3).

Run from the UE5 editor (requires the *Python Editor Script Plugin*):

    py "<ProjectDir>/Plugins/sagad/Content/Python/build_wpo_material.py"

Creates ``/sagad/Materials/M_SagadWPO`` -- the single shared material every scattered
instance uses. It combines the **baked vertex colors** (the shared mask: R=bend,
G=noise, B=scale, A=mobility) with the **Per-Instance Custom Data** gains written by
ASagadScatterActor to drive World Position Offset, so K instances deform differently
from one mesh file.

Per-Instance Custom Data layout (must match the diffusion spec / the scatter actor):
    [0] bend_gain   [1] noise_gain   [2] scale_gain   [3] base_band (reserved)

WPO accumulator (an in-shader approximation of the §2.3 sandbox):
    bend   = BendAxis     * (BendStrength  * R * bend_gain  * heightZ)
    noise  = VertexNormal * (NoiseStrength * G * noise_gain * Noise(worldPos))
    scale  = (worldPos - objPos) * (ScaleStrength * B * scale_gain)
    WPO    = (bend + noise + scale) * A          # A = mobility = 1 - fixed

Strength scalars are ScalarParameters so the material can be tuned to the asset's
real UE size (the baked mesh is unit-bounding-sphere normalized -- see the bake
sidecar JSON ``transform``).
"""

import unreal

PKG_PATH = "/sagad/Materials"
MAT_NAME = "M_SagadWPO"

MEL = unreal.MaterialEditingLibrary
_assets = unreal.AssetToolsHelpers.get_asset_tools()


def _make_material():
    full = "{}/{}".format(PKG_PATH, MAT_NAME)
    if unreal.EditorAssetLibrary.does_asset_exist(full):
        unreal.EditorAssetLibrary.delete_asset(full)
    mat = _assets.create_asset(MAT_NAME, PKG_PATH, unreal.Material,
                               unreal.MaterialFactoryNew())
    # HISM custom data requires the instanced-static-mesh usage flag.
    mat.set_editor_property("used_with_instanced_static_meshes", True)
    return mat


def _expr(mat, cls, x, y):
    return MEL.create_material_expression(mat, cls, x, y)


def _scalar(mat, name, value, x, y):
    e = _expr(mat, unreal.MaterialExpressionScalarParameter, x, y)
    e.set_editor_property("parameter_name", name)
    e.set_editor_property("default_value", value)
    return e


def _picd(mat, index, x, y):
    """One Per-Instance Custom Data float (the gain written by the scatter actor)."""
    e = _expr(mat, unreal.MaterialExpressionPerInstanceCustomData, x, y)
    e.set_editor_property("data_index", index)
    e.set_editor_property("default_value", 0.0)
    return e


def _vec3(mat, vec, x, y):
    e = _expr(mat, unreal.MaterialExpressionConstant3Vector, x, y)
    e.set_editor_property("constant", unreal.LinearColor(vec[0], vec[1], vec[2], 0.0))
    return e


def _mul(mat, a, ao, b, bo, x, y):
    e = _expr(mat, unreal.MaterialExpressionMultiply, x, y)
    MEL.connect_material_expressions(a, ao, e, "A")
    MEL.connect_material_expressions(b, bo, e, "B")
    return e


def _add(mat, a, b, x, y):
    e = _expr(mat, unreal.MaterialExpressionAdd, x, y)
    MEL.connect_material_expressions(a, "", e, "A")
    MEL.connect_material_expressions(b, "", e, "B")
    return e


def build():
    mat = _make_material()

    # -- shared inputs ----------------------------------------------------------
    vcol = _expr(mat, unreal.MaterialExpressionVertexColor, -1600, 0)
    normal = _expr(mat, unreal.MaterialExpressionVertexNormalWS, -1600, 300)
    world = _expr(mat, unreal.MaterialExpressionWorldPosition, -1600, 500)
    objpos = _expr(mat, unreal.MaterialExpressionObjectPositionWS, -1600, 650)

    bend_gain = _picd(mat, 0, -1600, -400)
    noise_gain = _picd(mat, 1, -1600, -300)
    scale_gain = _picd(mat, 2, -1600, -200)

    bend_str = _scalar(mat, "BendStrength", 60.0, -1600, -700)
    noise_str = _scalar(mat, "NoiseStrength", 5.0, -1600, -600)
    scale_str = _scalar(mat, "ScaleStrength", 15.0, -1600, -500)
    # Noise spatial frequency is a compile-time property of the Noise node (not an
    # input pin), so it can't be a runtime ScalarParameter -- set it as a constant.
    NOISE_FREQUENCY = 0.05
    bend_axis = _vec3(mat, (1.0, 0.0, 0.0), -1600, 800)   # horizontal lean direction

    # -- BEND: BendAxis * (BendStrength * R * bend_gain * heightZ) ---------------
    height_z = _expr(mat, unreal.MaterialExpressionComponentMask, -1300, 500)
    height_z.set_editor_property("r", False)
    height_z.set_editor_property("g", False)
    height_z.set_editor_property("b", True)    # Z (height above pivot, world scale)
    height_z.set_editor_property("a", False)
    MEL.connect_material_expressions(world, "", height_z, "")  # approx height; pivot at ground
    b1 = _mul(mat, bend_str, "", vcol, "R", -1100, -700)
    b2 = _mul(mat, b1, "", bend_gain, "", -950, -700)
    b3 = _mul(mat, b2, "", height_z, "", -800, -700)
    bend_off = _mul(mat, bend_axis, "", b3, "", -650, -700)

    # -- NOISE: VertexNormal * (NoiseStrength * G * noise_gain * Noise) ----------
    noise = _expr(mat, unreal.MaterialExpressionNoise, -1300, 300)
    noise.set_editor_property("scale", NOISE_FREQUENCY)
    MEL.connect_material_expressions(world, "", noise, "Position")
    n1 = _mul(mat, noise_str, "", vcol, "G", -1100, -300)
    n2 = _mul(mat, n1, "", noise_gain, "", -950, -300)
    n3 = _mul(mat, n2, "", noise, "", -800, -300)
    noise_off = _mul(mat, normal, "", n3, "", -650, -300)

    # -- SCALE: (worldPos - objPos) * (ScaleStrength * B * scale_gain) ----------
    radial = _expr(mat, unreal.MaterialExpressionSubtract, -1100, 600)
    MEL.connect_material_expressions(world, "", radial, "A")
    MEL.connect_material_expressions(objpos, "", radial, "B")
    s1 = _mul(mat, scale_str, "", vcol, "B", -1100, 100)
    s2 = _mul(mat, s1, "", scale_gain, "", -950, 100)
    scale_off = _mul(mat, radial, "", s2, "", -650, 100)

    # -- composite, gated by mobility (baked Alpha) -----------------------------
    sum1 = _add(mat, bend_off, noise_off, -450, -400)
    composite = _add(mat, sum1, scale_off, -300, -200)
    wpo = _mul(mat, composite, "", vcol, "A", -150, -200)   # * mobility

    MEL.connect_material_property(wpo, "", unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET)

    MEL.recompile_material(mat)
    unreal.EditorAssetLibrary.save_asset("{}/{}".format(PKG_PATH, MAT_NAME))
    unreal.log("Built {}/{}".format(PKG_PATH, MAT_NAME))


if __name__ == "__main__":
    build()
