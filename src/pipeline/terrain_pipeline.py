"""
Terrain Pipeline
================
Posloupnost:
  1.  Stažení DEM (OpenTopography)
  2.  Terrain grid  – bilinear upsample (z config.TERRAIN_UPSAMPLE)
  3.  OSM silnice → 2D union polygon (Shapely)
  4.  Z-burn: terrain vrcholy uvnitř road polygonu -= ROAD_BURN_DEPTH
      → silnice "sedí v krajině", kopíruje terén, je mírně zapuštěna
  5.  Road textura (PNG, soft-edge Gaussian blur) → UV na terrain meshi
  6.  Terrain OBJ s UV souřadnicemi + MTL soubor
  7.  Buildings OBJ – vytlačené budovy, základna na min-vertex Z terénu
      (Z-burn je zahrnut → budovy stojí na správné výšce)

Žádný separátní road OBJ → žádné Z-fighting, Draco komprese funguje.

Konfigurace: edituj /config.py v kořeni projektu.
"""

import os
import sys


class TerrainPipeline:

    def __init__(self, client, converter, osm_client=None,
                 road_union=None, road_texture=None,
                 building_extruder=None, cfg=None):
        self.client            = client
        self.converter         = converter
        self.osm_client        = osm_client
        self.road_union        = road_union
        self.road_texture      = road_texture
        self.building_extruder = building_extruder
        self.cfg               = cfg

    # ------------------------------------------------------------------ #

    def run_pipeline(self, bbox: tuple, output_base_name: str,
                     output_dir: str) -> dict:
        cfg = self.cfg

        print(f"\n{'='*60}")
        print(f"  Pipeline: '{output_base_name}'  bbox={bbox}")
        print(f"  Upsample: {cfg.TERRAIN_UPSAMPLE}×  "
              f"Road union: {cfg.ROAD_USE_2D_UNION}")
        print(f"{'='*60}")

        result: dict[str, str | None] = {
            "terrain": None, "texture": None, "buildings": None
        }

        # ── 1. DEM ────────────────────────────────────────────────────────
        print("\n[1/6] Stahuji DEM...")
        tif_path = self.client.get_dem(bbox)
        if not tif_path or not os.path.exists(tif_path):
            print("  ✗ Nepodařilo se stáhnout DEM.")
            return result
        print(f"  ✓ {tif_path}")

        # ── 2. Terrain grid ───────────────────────────────────────────────
        print(f"\n[2/6] Terrain grid ({cfg.TERRAIN_UPSAMPLE}× bilinear)...")
        try:
            z_grid, meta = self.converter.build_grid(
                tif_path, upsample_factor=cfg.TERRAIN_UPSAMPLE
            )
        except Exception as e:
            print(f"  ✗ {e}")
            return result

        # ── 3. OSM silnice → 2D union polygon ─────────────────────────────
        road_poly = None
        if self.osm_client and self.road_union:
            print("\n[3/6] OSM silnice → union polygon...")
            try:
                highways = self.osm_client.get_highways(bbox)
                print(f"  ✓ {len(highways)} silnic")
                road_poly = self.road_union._build_union(highways, meta)
                if road_poly and not road_poly.is_empty:
                    simp = getattr(cfg, "ROAD_BOUNDARY_SIMPLIFY", 0.15)
                    if simp > 0:
                        road_poly = road_poly.simplify(simp, preserve_topology=True)
                    print(f"  ✓ road union OK")
            except Exception as e:
                print(f"  ✗ road union selhal: {e}")

        # ── 4. Z-burn: terrain vrcholy uvnitř road polygonu ─────────────
        burn_depth = getattr(cfg, "ROAD_BURN_DEPTH", 0.08)
        z_burned   = z_grid   # výchozí: beze změny
        if road_poly and not road_poly.is_empty:
            print(f"\n[4/6] Z-burn (hloubka {burn_depth*100:.0f} cm)...")
            try:
                z_burned = self.converter.apply_road_burn(
                    z_grid, meta, road_poly, depth=burn_depth
                )
            except Exception as e:
                print(f"  ✗ Z-burn selhal: {e}")

        # ── 5. Road textura (PNG + soft edge) ─────────────────────────────
        texture_path = None
        if road_poly and not road_poly.is_empty and self.road_texture:
            print(f"\n[5/6] Road textura...")
            try:
                texture_path = self.road_texture.generate(
                    road_poly, meta,
                    output_dir=output_dir,
                    base_name=output_base_name,
                )
                result["texture"] = texture_path
            except Exception as e:
                print(f"  ✗ Textura selhala: {e}")

        # ── 6. Terrain OBJ (z Z-burned gridem + UV) ───────────────────────
        print(f"\n[6/6] Terrain OBJ...")
        try:
            terrain_path = self.converter.write_obj(
                z_burned, meta,
                obj_name=f"{output_base_name}_terrain",
                output_dir=output_dir,
                texture_file=texture_path,
            )
            result["terrain"] = terrain_path
        except Exception as e:
            print(f"  ✗ {e}")
            return result

        # ── 7. Buildings ──────────────────────────────────────────────────
        if self.osm_client and self.building_extruder:
            buildings = self.osm_client.get_buildings(bbox)
            if buildings:
                print(f"\n[7] Budovy ({len(buildings)})...")
                try:
                    # Budovy vzorkují z_burned → základna na Z-burned terénu
                    bld_path = self.building_extruder.extrude_buildings(
                        buildings, z_burned, meta,
                        obj_name=output_base_name,
                        output_dir=output_dir,
                    )
                    if bld_path:
                        result["buildings"] = bld_path
                except Exception as e:
                    print(f"  ✗ {e}")

        # ── Summary ───────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("  HOTOVO")
        for k, v in result.items():
            if v and os.path.exists(v):
                kb = os.path.getsize(v) / 1024
                unit = "MB" if kb > 1024 else "KB"
                size = kb / 1024 if kb > 1024 else kb
                print(f"    {k:12s}: {v}  ({size:.1f} {unit})")
            else:
                print(f"    {k:12s}: —")
        print(f"{'='*60}\n")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Spuštění jako standalone skript
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, ".")

    import config as cfg  # noqa: E402  (root config.py)

    from src.extraction.opentopography_client import OpenTopographyClient
    from src.extraction.osm_client             import OsmClient
    from src.conversion.terrain_converter      import TerrainConverter
    from src.conversion.road_union             import RoadUnion
    from src.conversion.road_texture           import RoadTexture
    from src.conversion.building_extruder      import BuildingExtruder

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    pipeline = TerrainPipeline(
        client            = OpenTopographyClient(),
        converter         = TerrainConverter(output_dir=cfg.OUTPUT_DIR),
        osm_client        = OsmClient(),
        road_union        = RoadUnion(cfg),
        road_texture      = RoadTexture(cfg),
        building_extruder = BuildingExtruder(output_dir=cfg.OUTPUT_DIR),
        cfg               = cfg,
    )

    pipeline.run_pipeline(cfg.BBOX, cfg.OUTPUT_NAME, cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
