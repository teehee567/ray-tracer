# Ray Tracer Acceleration Structures

A catalog of acceleration structures worth implementing for learning. Your current implementation uses a **SAH BVH with uniform spatial candidates** (8 candidates per axis, 3 axes). Everything below is something you can add, swap in, or compare against it.

---

## What You Already Have

- **BVH with SAH (Surface Area Heuristic)** — splits by evaluating `N_left * SA_left + N_right * SA_right` across 8 uniform positions per axis.
- **AABB traversal** — slab method, iterative stack-based, sorted by distance.
- **Triangle intersection** — Möller-Trumbore.

---

## 1. BVH Construction Variants

These all share the same BVH node layout and traversal — only the splitting strategy changes.

### 1a. Midpoint Split
Split each node at the centroid of the bounding box along the longest axis. Fast to build (O(n log n)), but quality is poor on clustered geometry. Good baseline to compare SAH against.

**Split position:** `(node_min[axis] + node_max[axis]) / 2`

### 1b. Median Split (Spatial Median)
Sort primitives by centroid along the longest axis, split at the median index. Guarantees balanced trees. Better than midpoint on uniform distributions, still worse than SAH on non-uniform ones.

**Split position:** sort by centroid, take `indices[n/2]`.

### 1c. SAH with Binned Candidates (what you have)
Divide the axis into N uniform bins, evaluate SAH cost at each bin boundary. You use 8 bins. Common values are 8–32. More bins = better quality, slower build.

### 1d. SAH with Centroid Binning (PBRT-style)
Instead of binning the spatial extent, bin the *centroid* extent only. Slightly better split quality on elongated geometry because bins are denser where primitives actually are.

```
centroid_min[axis] = min of all triangle centroids on this axis
centroid_max[axis] = max of all triangle centroids on this axis
bin_index = (centroid[axis] - centroid_min) / (centroid_max - centroid_min) * NUM_BINS
```

### 1e. Full SAH (Exact, O(n log n))
Sort primitives by centroid, then sweep left-to-right and right-to-left to compute prefix/suffix areas in O(n). Evaluate SAH at every possible split. Produces the best possible binary split, useful as ground-truth comparison. Only practical at build-time, not real-time.

### 1f. Object Median (Equal Counts)
Like median split but guarantees exactly n/2 on each side. Use `nth_element`-equivalent (partial sort). Decent quality, O(n) per level.

### 1g. Linear BVH (LBVH / Morton Codes)
Compute a Morton code for each primitive centroid (interleave the bits of x, y, z integer coordinates). Sort by Morton code, then split at the highest differing bit. O(n log n) sort, then O(n) hierarchy. Very GPU-friendly. Quality is lower than SAH, but construction is 10-50x faster — useful for dynamic scenes.

**Steps:**
1. Normalize all centroids into [0, 1]^3.
2. Multiply by 1024 (10-bit per axis), cast to integer.
3. Interleave bits: x₀ y₀ z₀ x₁ y₁ z₁ … (30-bit Morton code).
4. Sort primitives by Morton code.
5. Recursively split at the highest bit where left and right differ (`clz(code[first] ^ code[last])`).
6. Fit AABBs bottom-up.

### 1h. HLBVH (Hierarchical LBVH)
Build the top N levels using Morton-code treelet splits (fast, parallel), then apply SAH within each treelet. Used in production GPU renderers — combines LBVH's parallelism with SAH's quality at the leaf level.

### 1i. Spatial Splits (SBVH)
Allow a triangle to appear in both the left and right child if it straddles the split plane. This avoids the "large triangle spanning many cells" problem where SAH is forced to put a triangle in an oversized bin. Costs extra memory (duplicated references) but reduces traversal steps significantly on geometry with large triangles.

**When to use:** scenes with large area-light planes, terrain, or architectural geometry.

### 1j. Agglomerative Clustering (ATRBVH / Ploc)
Build bottom-up: start with one leaf per primitive, repeatedly merge the pair with the lowest SAH cost. Higher quality than top-down SAH on average but slower to build. PLOC (Parallel Locally-Ordered Clustering) is the modern GPU-parallel variant.

---

## 2. BVH Traversal Variants

### 2a. Stackless Traversal (Rope BVH)
Precompute "ropes" — pointers to the next subtree to visit when backtracking. Eliminates the stack entirely. Useful on GPUs where per-thread stacks are expensive in register pressure.

### 2b. Ordered Traversal (what you have, partially)
You already sort left/right by distance and push furthest first. Full ordered traversal also prunes the far child early using the current best hit distance `t_max`.

### 2c. Packet Traversal (SIMD)
Process 4 or 8 rays simultaneously through the same BVH. When all rays miss a node, skip it; when all hit, traverse both children. Effective on coherent rays (primary rays, shadow rays to a single light). Less effective on secondary/diffuse rays.

### 2d. Wide BVH (4-ary, 8-ary)
Replace the binary tree with a 4-wide or 8-wide tree. Each node has up to 8 children. AABB test 8 children at once with SIMD. Reduces tree depth, improves cache behavior. Embree and OptiX use 4-wide or 8-wide BVHs. Requires collapsing binary BVH nodes or building wide directly.

---

## 3. Alternative Space-Partitioning Structures

These are fundamentally different from BVH — they partition *space* rather than primitives.

### 3a. Uniform Grid
Divide the scene AABB into a regular N×N×N grid. Each cell holds a list of overlapping triangles. Traverse with 3D DDA (same algorithm as voxel ray casting). O(1) build per primitive, O(N) traversal worst case.

**Best for:** uniformly distributed scenes (point clouds, particle systems). Bad for scenes with mixed scales.

**Grid DDA traversal:** Amanatides & Woo (1987) — compute `t_delta` and `t_max` per axis, step the axis with smallest `t_max`.

### 3b. KD-Tree
Recursively split space with an axis-aligned plane (not primitives — the plane can cut through a triangle). Triangles straddling the plane go into both children. Historically outperformed BVH on static scenes; now generally matched or beaten by good SAH BVH with spatial splits.

**Key difference from BVH:** splits space, not primitive sets. A triangle can appear in multiple leaves.

**Traversal:** ordered — compute entry/exit `t` for the split plane, traverse near child first, then far child only if ray reaches it.

**SAH for KD-trees:** `cost = C_trav + (SA_left/SA_parent) * N_left * C_isect + (SA_right/SA_parent) * N_right * C_isect`. Terminate when cost exceeds leaf cost.

### 3c. Octree
Recursively subdivide the scene cube into 8 equal children. Simpler than KD-tree (no split plane choice), naturally adapts to 3D geometry, but split quality is fixed and can't be optimized. Good for sparse voxel scenes or SDF acceleration.

### 3d. BSP Tree (Binary Space Partitioning)
Like KD-tree but the splitting plane can be *any* plane, not just axis-aligned. More general, much more expensive to build. Rarely used in ray tracing now; common in old rasterization-based visibility (Quake, Doom).

---

## 4. Grid + Hierarchy Hybrids

### 4a. Hierarchical Grid (Hgrid)
Multi-resolution grid: coarse grid at the top level, finer grids within cells that contain many primitives. Handles mixed-scale scenes better than a uniform grid.

### 4b. BVH + Grid Leaf
Build a BVH down to a certain depth, then use a uniform grid within each leaf instead of a list. Useful when leaf triangles are spatially scattered.

---

## 5. Implicit / Analytic Accelerators

### 5a. Bounding Sphere
Cheap reject before triangle test: compute `dot(ray.origin - sphere.center, ray.direction)` and check discriminant. Useful as a pre-pass before expensive intersection tests or BVH traversal for entire objects.

### 5b. OBB (Oriented Bounding Box)
Tighter than AABB for rotated objects. Transform ray into the OBB's local space, then treat as AABB intersection. More expensive per test (~3× AABB), but can save traversal steps for long thin objects at an angle.

### 5c. Convex Hull BVH
Replace AABB nodes with convex hulls for tighter fits. Rarely practical — convex hull intersection is much more expensive than AABB — but useful for very large, few, convex objects.

---

## 6. Distance Field Structures

### 6a. SDF (Signed Distance Field) Grid
Precompute the signed distance to the nearest surface at every voxel. Traverse with sphere tracing: advance the ray by the SDF value at the current point (guaranteed safe step). Exact for analytic SDFs, approximate for baked grids.

**Sphere tracing:** `p += ray.dir * sdf(p)` until `sdf(p) < epsilon`.

### 6b. Sparse Voxel Octree (SVO)
Store only non-empty voxels in an octree. Efficient for volumetric or voxel-style scenes. Ray traversal visits nodes that overlap the ray, skipping empty space. Used in "Efficient Sparse Voxel Octrees" (Laine & Karras, 2010).

### 6c. Sparse Voxel DAG (SVDAG)
Extend SVO by merging identical subtrees into a DAG. Enormous compression for repetitive geometry (terrain, foliage). Read-only — no easy updates.

---

## 7. Light-Specific Accelerators

### 7a. Light BVH / Light Tree
A BVH built over light sources for importance sampling — traverse to find the most important light for a given shading point. Used in production renderers (Manuka, Arnold). Nodes store total power and bounding cone of emission directions.

### 7b. Alias Table for Area Lights
Precompute an alias table over triangle lights weighted by area × emissivity. O(1) light sampling. Combine with per-triangle power to importance sample emissive geometry.

---

## 8. Scene-Level Structures

### 8a. Two-Level BVH (TLAS + BLAS)
- **BLAS** (Bottom-Level AS): one BVH per mesh in object space.
- **TLAS** (Top-Level AS): BVH over transformed BLAS instances.

Each instance stores a transform matrix. The ray is transformed into object space before traversing the BLAS. This is exactly what Vulkan/DXR hardware ray tracing exposes. Enables instancing: one mesh BVH reused for N copies.

### 8b. Per-Material BVH
Separate BVH per material. Useful for alpha-masked geometry: traverse the opaque BVH first, only fall back to the masked BVH if needed.

---

## Implementation Priority Suggestions

If you're going through these for experience, a reasonable order:

1. **Midpoint split** and **median split** — trivial, good SAH comparison baselines.
2. **Centroid binning** (1d) — one small change from what you have, measurable improvement.
3. **Full exact SAH** (1e) — see the quality ceiling.
4. **Morton code / LBVH** (1g) — completely different approach, teaches Morton coding.
5. **KD-tree with SAH** (3b) — fundamentally different structure, good contrast.
6. **Uniform grid + DDA** (3a) — simple, teaches 3D DDA traversal.
7. **Two-level BVH** (8a) — teaches instancing, directly mirrors GPU RT APIs.
8. **Wide BVH 4-ary** (2d) — teaches SIMD-friendly layout and node collapse.
9. **Spatial splits / SBVH** (1i) — teaches reference duplication for better splits.
10. **LBVH → HLBVH** (1h) — production technique, prepares for GPU BVH.

---

## Key Papers

| Structure | Paper |
|---|---|
| SAH BVH | Goldsmith & Salmon 1987; MacDonald & Booth 1990 |
| KD-tree SAH | Wald & Havran 2006, "On building fast kd-trees" |
| LBVH | Lauterbach et al. 2009 |
| HLBVH | Pantaleoni & Luebke 2010 |
| SBVH | Stich et al. 2009 |
| PLOC | Meister & Bittner 2018 |
| SVO | Laine & Karras 2010 |
| SVDAG | Kampe et al. 2013 |
| Light BVH | Conty Estevez & Kulla 2018 ("Importance Sampling of Many Lights") |
| Wide BVH | Wald et al. 2008, "Ray Tracing Deformable Scenes Using Dynamic BVH" |
