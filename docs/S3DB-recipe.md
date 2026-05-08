# Simple 3D Buildings (S3DB) — Implementation Recipe

Source: https://wiki.openstreetmap.org/wiki/Simple3DBuildingsV1

---

## Height tags (all building types)

| Tag | Meaning |
|-----|---------|
| `height` / `building:height` | Total height from ground to roof top (metres). |
| `min_height` / `building:min_height` | Height of the bottom face above ground (for overhangs, bridges, columns). `min_height=10, height=13` → 3 m tall structure floating 10 m up. |
| `building:levels` / `levels` | Floor count above ground. 1 level ≈ 3.0 m. |
| `building:min_level` / `min_level` | Lowest floor index (like min_height but in levels). |
| `roof:height` | Vertical extent of the roof alone (from eave to ridge/apex). |
| `roof:levels` | Floors inside the roof (add to building:levels for total). |
| `roof:angle` | Slope angle in degrees (alternative to roof:height). |

### Height derivation order
```
total_height =
    height || building:height
    || (building:levels + roof:levels) * 3.0
    || 10.0   (default)

roof_height  =
    roof:height
    || roof:levels * 3.0
    || roof:angle → 0.5 * half_width * tan(angle)
    || 1.0   (default, only for non-flat shapes)

wall_height  = total_height − min_height − roof_height   (≥ 0)
plate_z      = terrain_anchor + min_height + wall_height
             = terrain_anchor + total_height − roof_height
```

---

## Roof shapes

All non-flat roof types that assume a **ridge** need a **rectangle** as their base.
Use the **Oriented Bounding Box (OBBox)** of the polygon for that rectangle.

### OBBox rules (per spec, section 3.5)

| Polygon | Rule |
|---------|------|
| 4 corners (simple rectangle) | Use vertices as-is. |
| 4 corners + co-linear nodes (< 0.10 m off edge) | Drop co-linear nodes, use the 4 corners. |
| > 4 corners (L, T, U, irregular …) | Compute minimum-area OBBox. Roof is built on the OBBox rectangle. |

The OBBox roof overhangs the actual polygon walls for irregular shapes.
**Clip operation** (optional but recommended): for each wall vertex compute the roof
surface height directly above it and use that as `top_z`.  This extends the walls
to meet the roof so there is no visible gap.

### Roof type table

| `roof:shape` | Description | Construction |
|---|---|---|
| `flat` | Flat polygon at `plate_z`. | Triangulate polygon. |
| `skillion` | Single slope. Low side = `plate_z`, high side = `plate_z + roof_height`. | Per-vertex z based on `roof:direction`. |
| `gabled` | Two slopes meeting at a ridge. Ridge parallel to OBBox long axis. | OBBox + clip + CDT. End inset = 0. |
| `hipped` | All four sides slope. Ridge shorter than building. | OBBox + clip + CDT. End inset ≈ 0.30. |
| `half-hipped` | Like hipped but only the short ends are hipped. | OBBox + clip + CDT. End inset ≈ 0.15. |
| `gambrel` | Two slopes per side (lower slope steeper). | Treat as gabled (approximation). |
| `mansard` | Two slopes on every side (lower steeper). | Treat as gabled (approximation). |
| `round` | Barrel-shaped curved roof. | Treat as gabled (approximation). |
| `pyramidal` | All sides converge to single apex at centroid. | Apex = centroid at `plate_z + roof_height`. |
| `dome` | Hemispherical, rising rings of n vertices. | Latitude rings + apex. |
| `onion` | Onion-shaped dome variant. | Same as dome, different height distribution. |

### Ridge direction (orientation)

1. Check `roof:direction` tag (compass bearing): direction **water flows** (downslope).
   Ridge is perpendicular to that direction.
2. Else check `roof:orientation=across`: ridge runs across the short axis (unusual).
3. Default: ridge along the **long axis** of the OBBox (`roof:orientation=along`).

---

## OBBox-based ridge construction (gabled / hipped)

```
1.  Compute OBBox → (cx, cy, half_len, half_wid, rdx, rdy, pdx, pdy)
    rdx,rdy = long-axis unit vector
    pdx,pdy = short-axis unit vector (perpendicular)
    half_len ≥ half_wid

2.  Apply roof:direction / roof:orientation override if present.

3.  Ridge span:
        inset   = 2 * half_len * end_inset          (0 for gabled)
        r1_rel  = −half_len + inset                 (from OBBox centre)
        r2_rel  = +half_len − inset
        if r1_rel ≥ r2_rel → fall back to pyramidal

4.  Per-vertex z (CLIP OPERATION):
    for each polygon vertex (vx, vy):
        t = (vx−cx)*rdx + (vy−cy)*rdy              (along-ridge offset from centre)
        d = |(vx−cx)*pdx + (vy−cy)*pdy|            (perp distance from ridge centreline)

        z_side = max(0, 1 − d / half_wid)
        z_hip1 = (t − (−half_len)) / inset         (or 1.0 when inset=0)
        z_hip2 = (half_len − t)    / inset
        factor = min(z_side, z_hip1, z_hip2)        (all three only for hipped)

        top_z[i] = plate_z + roof_height * max(0, factor)

5.  Ridge endpoints:
    gabled:   find where ridge centreline intersects polygon boundary → R1, R2
    hipped:   interior points:
              R1 = centre + r1_rel * (rdx, rdy)
              R2 = centre + r2_rel * (rdx, rdy)
    z(R1) = z(R2) = plate_z + roof_height

6.  Triangulate roof cap:
    a. For gabled: insert R1,R2 into polygon boundary at correct edges, run CDT.
    b. For hipped: add R1,R2 as interior vertices in PSLG, run CDT.
    Indices 0..n-1 = top ring;  n = R1;  n+1 = R2.
```

---

## Building relation handling

- `relation[type=building]` members with role `outline` → suppressed when parts present.
- `relation[type=building]` members with role `part` → treated as building:parts.
- `way[building:part]` ways that are outer rings of a building relation → **do not render twice**.
- `_suppress_outer_shells`: if a `building=yes` outline overlaps any `building:part`
  by ≥ 15 % of the part area → suppress the outline.

---

## Terrain anchoring

- `anchor_z` = terrain elevation at the **footprint centroid** (consistent across parts).
- Safety clamp (only when `wall_h > 0`): `anchor_z = max(anchor_z, max(corner_z) − wall_h * 0.9)`.
- Ground-level (`min_height = 0`): base follows terrain per vertex.
- Elevated (`min_height > 0`): flat base at `anchor_z + min_height`.

---

## Polygon orientation

OSM ways may be clockwise or counter-clockwise. **Normalise to CCW** before ALL
geometry generation so face normals point outward.
