# Analysis: Tiles Generation Pipeline

## Current State
The current tile generation logic is fragmented across numerous Python scripts located in `svarog/scripts/terrain/`. These scripts perform diverse, single-purpose tasks such as:
- `generate-all-layers-glb-tiles.py`: The primary (but monolithic) orchestration script.
- `generate-road-glb-tiles.py`: Specialized road texture/mesh generation.
- `process-single-tile-with-terrain-anchored-buildings.py`: Building placement logic.
- `optimize_glb.py`: Post-processing and Draco compression.

While functional, this approach lacks a unified pipeline structure, versioned contracts, and a clear distinction between "data extraction" and "asset generation."

## Goal for `svarog-engine`
Transform these scripts into a cohesive, modular, and contract-driven pipeline.

### 1. Core Architecture: The "Transformer" Pattern
The engine should implement a pipeline where each stage is a discrete, testable module:
`Raw Data (OSM/DEM) → Stage: Extraction → Stage: Geometry Processing → Stage: Mesh Generation → Stage: Optimization → Final Asset (GLB + Manifest)`

### 2. Key Responsibilities of `svarog-engine`
- **Geometry Engine:** Handle polygon clipping, building extrusion, and road stamping (implementing logic from the existing `.py` scripts).
- **NavMesh Factory:** A dedicated module that processes generated geometry to produce NavMesh graphs (using `recast-navigation` or similar).
- **Contract Enforcement:** Every generation run must produce a `tile-manifest.json` that validates against `svarog-contracts`.
- **Asset Optimization:** Integrated Draco compression and GLB simplification as a mandatory final step.

### 3. Refactoring Roadmap

#### Phase 1: Extraction & Structure (Short Term)
- Setup the `svarog-engine` project structure (e.g., using Python or Node.js/TypeScript).
- Port the core "Tile Extraction" logic from `generate-all-layers-glb-tiles.py`.
- Define the first "Internal Contract" for the pipeline's intermediate data format.

#### Phase 2: Modularization (Medium Term)
- Extract "Road Generation" and "Building Anchoring" into independent, plugin-like modules.
- Implement a centralized "Task Runner" that can orchestrate single-tile or batch-tile generation.
- Integrate `svarog-contracts` to ensure the output `tile-manifest.json` is always valid.

#### Phase 3: NavMesh & Advanced Features (Long Term)
- Implement the `NavMesh Factory` using the geometry produced in Phase 2.
- Add support for advanced terrain layers (e.g., seasonal changes, water levels) by adding new stages to the pipeline.

## Critical Dependencies
- **`svarog-contracts`**: Essential for defining the schema of the output manifests.
- **`svarog` (Legacy)**: Used as a reference for the expected input/output properties of the tiles.
