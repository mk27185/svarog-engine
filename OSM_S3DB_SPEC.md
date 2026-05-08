# OSM Simple 3D Buildings — Implementation Specification

Source: https://wiki.openstreetmap.org/wiki/Simple_3D_Buildings  
Last wiki revision used: 19 April 2026

---

## 1. Data model

### 1.1 Building outlines (`building=*`)

- Represent the **union** of all parts of a building (the footprint/shadow on the ground).
- May be a **closed way** or a **multipolygon relation**.
- Carry building-level attributes: name, address, overall height, etc.
- **When any `building:part=*` area exists that lies within this outline, the outline is NOT
  rendered in 3D** — only the parts are rendered. The outline exists for 2D-map backward
  compatibility only.

### 1.2 Building parts (`building:part=*`)

- Describe sections of a building with **different height or other physical attributes**.
- `building:part=yes` is the most common value; any `building=*` value is also valid.
- The **entire building outline should be filled** with `building:part` areas.
- Areas may overlap in 2D but 3D volumes should not share faces.

### 1.3 Building relations (`type=building`)

- Optional; used when parts overhang the outline or the structure is complex.
- Members: `role=outline` for `building=*`, `role=part` for `building:part=*`.
- When a `type=building` relation exists, parts may lie ANYWHERE relative to the outline
  (inside, outside, overlapping).
- Without such a relation, treat all `building:part` areas whose **centroid** (or majority) lies
  within the `building=*` outline as parts of that building.

---

## 2. Height tags

### 2.1 Absolute heights (metres above ground)

| Tag | Meaning |
|-----|---------|
| `height=*` | Distance from **lowest ground contact** to the **top of the roof** (excludes antennas, spires). This is the TOTAL height including walls AND roof. |
| `min_height=*` | Height at which the **bottom** of the structure starts above ground. E.g. a bridge 3 m tall whose underside is 10 m up: `min_height=10`, `height=13`. |

### 2.2 Level-based heights (fallback when no metric `height`)

| Tag | Meaning |
|-----|---------|
| `building:levels=*` | Number of floors **above ground**, NOT counting roof-space levels. |
| `building:min_level=*` | Floors skipped at the bottom (analogous to `min_height`). |
| `roof:levels=*` | Floors within the roof space (not counted in `building:levels`). |

**Level-to-metre conversion: 1 level = 3.0 m** (convention used by OSMBuildings / OSM2World).

### 2.3 Roof height

| Tag | Meaning |
|-----|---------|
| `roof:height=*` | Height of the roof section in metres. Façade height = `height − roof:height`. |
| `roof:angle=*` | Inclination of the roof sides in **degrees** (0–90). Alternative to `roof:height`. When given, `roof_height = tan(roof:angle°) × max_distance_from_eave_to_ridge`. |

**There is no `building:height` tag.** Façade height is always derived:
`façade_height = height − roof:height`.

### 2.4 Height derivation rules (implementation)

When `height=*` is present (explicit metres):

```
total_h   = float(height)
roof_h    = parse_roof_height(tags, total_h)   # see §3 priority
wall_h    = total_h − min_height − roof_h
plate_z   = ground_z + min_height + wall_h     # = ground_z + total_h − roof_h
roof_top_z = plate_z + roof_h                  # = ground_z + total_h  ✓
```

When **only `building:levels`** is present (no `height` tag):

```
wall_h    = building:levels × 3.0              # levels give WALL height
roof_h    = parse_roof_height(tags, ∞)         # roof is ADDITIONAL on top
plate_z   = ground_z + min_height + wall_h
roof_top_z = plate_z + roof_h
```

When **neither** is present: use a default wall height (e.g. 10 m) plus default roof height.

---

## 3. Roof height derivation (priority order)

1. `roof:height=*` — explicit metres → use directly, clamped to `total_h`
2. `roof:levels=*` — levels × 3 m; **`roof:levels=0` means no habitable roof space,
   but does NOT mean the roof is physically flat** → if value > 0, use it; else fall through
3. `roof:angle=*` — compute `tan(angle°) × max_eave_to_ridge_distance`
   (only valid once the ridge geometry is known; applied inside the ridged-roof builder)
4. **Default** — 3 m for any non-flat `roof:shape`; 0 m for `flat` / missing shape

---

## 4. Roof shapes (standardised S3DB values)

| Value | Description |
|-------|-------------|
| `flat` | No slope; top is a flat horizontal polygon. |
| `skillion` | Single sloped plane; one side high, opposite side low. |
| `gabled` | Two slopes meeting at a central **ridge** running along the building length. Gable ends are **vertical triangles**. |
| `hipped` | Four slopes all meeting at the ridge; **no vertical gable ends**. The two short ends are triangular slopes, not vertical. |
| `half-hipped` | Gabled with the gable ends **partially hipped** (clipped). |
| `pyramidal` | All sides meet at a single **apex** above the centroid. |
| `gambrel` | Double-slope on each side (like a barn): lower steep section + upper shallower section. |
| `mansard` | Like gambrel but all four sides (steep outer walls + nearly-flat upper section). |
| `dome` | Hemispherical cap. |
| `onion` | Bulging then tapering dome (wider than hemisphere near the base). |
| `round` | Cylindrical/barrel vault along the ridge axis. |
| `saltbox` | Asymmetric gable: ridge offset toward one side; one slope is longer and steeper. |

---

## 5. Ridge direction (for gabled, hipped, half-hipped, gambrel, mansard, saltbox, round)

### 5.1 Tag priority

1. **`roof:direction=*`** — compass bearing (0=N, 90=E, 180=S, 270=W) toward which the
   **main face of the roof looks** (= the downslope direction).  
   Ridge is **perpendicular**: `ridge_bearing = roof:direction + 90°`

2. **`roof:ridge:direction=*`** — compass bearing the **ridge itself runs**.  
   Used directly without rotation.

3. **Auto-detect from Oriented Bounding Box (OBBox)**:
   - `roof:orientation=along` (default): ridge runs along the **long axis** of the OBBox.
   - `roof:orientation=across`: ridge runs along the **short axis** of the OBBox.

### 5.2 OBBox calculation

Use **rotating calipers** over polygon edges to find the minimum-area bounding rectangle.
The long axis of that rectangle gives the default ridge direction.

---

## 6. Skillion roof

`roof:direction` = compass bearing the slope **faces** (downslope direction).  
Low eave: vertices whose projection onto the slope vector is smallest → stay at `plate_z`.  
High eave: vertices with largest projection → rise to `plate_z + roof_height`.  
All intermediate vertices interpolated linearly.

---

## 7. Outer-shell suppression rule

**Spec rule**: "When a building has any `building:part=*` areas, the building outline is
not considered for 3D rendering."

**Implementation**: suppress a `building=*` outline if the intersection area between
the outline polygon and any `building:part` polygon exceeds **15 % of the part's own area**.
(If a part is substantially inside the outline, the outline is suppressed.)

---

## 8. Multipolygon relations (buildings mapped as OSM relations)

A building or building:part that is a `type=multipolygon` relation in OSM has its perimeter
defined by **multiple ways** (outer ring members) that must be **assembled** into one or more
closed rings by chaining matching endpoints.

Algorithm:
1. Collect all way segments for `role=outer` members.
2. Chain them by matching endpoint coordinates until a closed ring is formed.
3. If multiple disconnected rings exist, produce one building entry per ring.
4. `role=inner` members are ignored for 3D building geometry (they define holes in the 2D
   footprint, not separate buildings; hole handling is a future enhancement).

---

## 9. Winding order and normals

- All polygon vertices must be in **CCW order** when viewed from above (positive Z = up).
- Wall faces: CCW winding when viewed from outside → outward-pointing normals.
- Roof faces: CCW winding when viewed from above → upward-pointing normals.

---

## 10. What is NOT in the spec (do not invent)

- No `building:height` tag exists.
- No automatic fallback from complex roof shapes to simpler ones (e.g. gabled → pyramid).
  If the geometry cannot be computed, fall back to `flat`, not to another shape.
- `roof:levels=0` does NOT mean flat — it means no habitable attic space.
  Non-flat `roof:shape` with `roof:levels=0` still gets the default roof height (3 m).
