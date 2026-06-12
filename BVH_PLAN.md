# Plan: Hyper-Optimised CPU-Build / GPU-Traverse BVH

Goal: a new accelerator (new module, `src/accelerators/bvh2.rs` or similar — `bvh.rs` stays
untouched) that produces the best overall balance of **build speed** and **tree quality**
for the existing pipeline:

- Built on CPU via `Accelerator::build(&mut Vec<Triangle>, &mut Vec<Material>) -> Vec<BvhNode>`
- Consumed on GPU by `intersects.comp`: stack traversal, 32-byte nodes, children adjacent
  at `idx` / `idx+1`, leaves = contiguous `[idx, idx+amt)` ranges of the flat triangle buffer
- Triangles are reordered in place; spheres ride along as `is_sphere` triangles, so all
  bounds must come from `Triangle::min_bound()/max_bound()`, never raw vertices

## Hard constraints (from the existing code)

1. **Output format is fixed** (until Phase 6): `Vec<BvhNode>` with the union layout in
   `bvh.rs`, root at index 0, sibling pairs adjacent, `amt != 0` ⇒ leaf.
2. **Flat triangle list**: leaf primitives must be contiguous after reordering
   (see `benches/bvh.rs` — the bench builds straight from `scene.components.triangles`).
3. **Stack depth**: shader stack is `MAX_BVH_DEPTH + 1`; keep tree depth ≤ that bound.
4. `BvhNode`'s fields are private to `bvh.rs`. The new builder needs either
   (a) small `pub fn leaf(min, max, first, count)` / `pub fn interior(min, max, left)`
   constructors added to `BvhNode` (one tiny additive change, no logic touched), or
   (b) its own `#[repr(C)]` twin struct + transmute. Prefer (a); decide at implementation
   time.

## Architecture choice

**Top-down binned SAH (Wald 2007) as the core, parallelised with rayon, with optional
spatial splits (SBVH) and an optional post-build reinsertion pass.** Rationale:

- Binned SAH is within 1–5% of full-sweep SAH quality at a fraction of the cost: O(N)
  per node instead of O(N × candidates).
- Morton/LBVH/PLOC builders are faster still but give measurably worse trees; since the
  build runs once on scene load and rays run every frame on GPU, quality is worth more
  than the last few ms of build time. Parallel binned SAH on Sponza-sized scenes
  (~260K tris) lands around 10–30 ms, which is "fast enough" territory.
- SBVH (Stich 2009) recovers another ~15–30% traversal performance on scenes with large
  thin triangles (Sponza's curtains/plants are the canonical example) at ~2–3× build cost
  and some triangle duplication — fits naturally as a quality dial, not a rewrite.

## Phase 0 — Scaffolding, measurement, correctness harness

Nothing gets optimised until it can be measured and validated.

1. New module `src/accelerators/bvh2.rs`, registered in `mod.rs`, implementing
   `Accelerator`. Start by passing the existing bench with a trivial median split so the
   plumbing is proven.
2. **Build-input SoA pre-pass**: one pass over triangles computing per-primitive
   `aabb_min`, `aabb_max`, `centroid` into tight arrays (32 B/prim vs touching the
   ~160 B `Triangle` repeatedly). Every later phase works only on these + an index array;
   triangles are permuted exactly once at the end (the current `apply_ordering` approach
   is correct, keep that pattern).
3. **Quality metrics** computed post-build and printed in the bench:
   - SAH cost of the tree (∑ interior `A_n/A_root · C_trav` + ∑ leaf `A_n/A_root · N · C_isect`)
   - node count, leaf count, max depth, avg/max leaf size
4. **Correctness tests**:
   - every triangle index appears exactly once across leaves (pre-SBVH), leaves cover
     `0..N` contiguously in node order
   - every leaf AABB contains its triangles' bounds (incl. sphere bounds)
   - interior AABB ⊇ children AABBs; `left`/`left+1` in range; depth ≤ `MAX_BVH_DEPTH`
   - a small CPU reference traversal: fire a few thousand rays through old BVH and new
     BVH over the same scene, assert identical nearest hits. This is the real safety net.
5. Extend `benches/bvh.rs` (or add `benches/bvh2.rs`) to bench both builders side by side
   with the same throughput setup.

## Phase 1 — Single-threaded binned SAH builder

The core algorithm; correct and already much faster + higher quality than the current one.

1. **Iterative build loop** with an explicit work stack (no recursion — avoids both stack
   overflow and makes Phase 2 parallelism trivial).
2. Per node: compute the **centroid bounds**; bin K=16 bins (compile-time const, tune in
   Phase 5) along all 3 axes in a **single pass** over the node's primitives (3×16 bins of
   `(aabb, count)` accumulated simultaneously — better than three passes for the fat-cache
   cost of the SoA arrays).
3. SAH evaluation via the standard **suffix-sweep**: right-to-left accumulate bin AABBs,
   then left-to-right sweep computing `cost = C_trav + (A_L·N_L + A_R·N_R) / A_parent · C_isect`.
   `C_trav = 1.0`, `C_isect = 1.0` to start; tune later (GPU triangle tests are relatively
   cheap vs. node fetches, so optimum likely has `C_isect < C_trav`).
4. **Termination**: create a leaf when `N ≤ min_leaf` (start: 2) or when best split cost ≥
   leaf cost `N · C_isect`, with a hard `max_leaf` (start: 8) that forces a median split
   instead of an oversized leaf. The current builder splits to 1 tri/leaf — that bloats
   node count and GPU node fetches; SAH-terminated leaves of 2–4 are typically optimal
   for this traversal style.
5. **Degenerate handling**: zero-extent centroid bounds on all axes → forced median/even
   split (never an unbounded leaf); NaN-free guarantee on bins (clamp bin index).
6. **In-place partition** of the index slice (like the current code), children allocated
   adjacently (`push` twice), node order therefore naturally groups siblings — matches
   the shader contract. Allocate `bvh_list` with `Vec::with_capacity(2 * N)` up front.

Expected on Sponza: build drops from the current multi-second sweep-SAH to roughly
~100 ms single-threaded, with *better* SAH cost (16 bins on centroid bounds beats 8
uniform slices of the node bounds, plus proper leaf termination).

## Phase 2 — Parallel build (rayon)

1. **Task-parallel top-down**: nodes with ≥ ~8–16K prims split as rayon `join` tasks
   (left/right subtrees in parallel); below the threshold, run the Phase 1 sequential
   path. The binning pass itself for the huge root node can additionally be
   chunk-parallel (per-thread bin sets, reduced at the end).
2. **Node allocation under parallelism**: sibling-adjacency makes a shared `Vec` +
   atomic counter awkward. Cleanest scheme: each parallel task gets its own node arena;
   parent records child placeholder; after the build, arenas are concatenated and child
   indices fixed up in one linear pass. (Alternative: pre-reserve `2·N_subtree` slots per
   task from an atomic bump allocator — zero fixup, slight memory waste. Decide by
   benchmarking; the fixup pass is trivially cheap either way.)
3. Determinism: same tree regardless of thread count (binning is deterministic; only
   memory placement varies, fixup normalises it). Keep a `RAYON_NUM_THREADS=1` test
   asserting identical SAH cost.

Expected: ~4–8× scaling on desktop core counts → **~10–30 ms for Sponza**, which beats
the bench's current numbers by orders of magnitude while improving quality.

## Phase 3 — Tree quality: spatial splits (SBVH), opt-in dial

The single biggest remaining quality lever for GPU traversal on this kind of content.

1. Implement Stich et al. 2009: at each node, alongside the binned **object split**,
   evaluate **spatial splits** (chopped binning: clip each triangle's AABB against bin
   planes, a triangle contributes to every bin it overlaps). Pick whichever split has
   lower SAH cost; restrict spatial splits to nodes where object-split overlap area is
   significant (`λ = A(overlap)/A(root) > α`, α ≈ 10⁻⁵) so most nodes pay nothing.
2. **Reference duplication and the flat-list constraint**: spatial splits put one
   triangle into multiple leaves. Because leaves must be contiguous ranges of the
   triangle buffer, duplicated references become **physically duplicated triangles**
   in the output `Vec<Triangle>`. That is legal here (the trait gets `&mut Vec` and the
   buffer is uploaded after build) and harmless for correctness — same surface, the
   nearest-hit is identical. Cap memory growth at ~25–30% extra refs (standard SBVH
   budget); when the budget is hit, fall back to object splits.
3. Triangle AABB clipping must use actual vertex/plane clipping (Sutherland–Hodgman on
   the axis plane), not AABB chopping, or quality gains evaporate. Spheres
   (`is_sphere == 1`) are never spatially split — object-split path only for them.
4. Make it a builder parameter: `SplitPolicy::Object` (default, fastest) vs
   `SplitPolicy::Spatial { alpha, budget }` for final-quality renders.

Expected: ~15–30% fewer ray–box/ray–tri tests on Sponza for ~2–3× the (parallel) build
time — still well under 100 ms.

## Phase 4 — GPU-side sympathy (no shader changes required)

Cheap CPU-side tweaks that target how `intersects.comp` actually consumes the tree:

1. **DFS node reordering post-pass**: re-lay nodes in depth-first order (children near
   parents) for better GPU cache hit rate on the `bvh.nodes` SSBO. Pure permutation +
   index fixup, O(N), measurable on divergent workloads.
2. **Child ordering**: place the larger-surface-area (more likely hit) child as the one
   that benefits from the shader's near-far ordering; the shader already distance-sorts,
   so this mostly matters for the degenerate equal-distance case — low priority, test it.
3. Verify depth bound: SAH+median-fallback trees on 260K tris come out ~25–35 deep,
   comfortably under `MAX_BVH_DEPTH = 64`; add the assert anyway.
4. Tighten leaf size against GPU reality: benchmark Sponza frame time (or
   rays/sec counter already present via `mesh_tests`) for `max_leaf` ∈ {2, 4, 6, 8} and
   pick empirically. CPU-side SAH cost is a proxy; the GPU number is the truth.

## Phase 5 — Build-speed micro-optimisation pass

Only after the above is correct and benchmarked; each item is measured in isolation.

1. SIMD binning: glam's `Vec3A`/SSE already vectorises the min/max; ensure the SoA
   arrays are 16-byte aligned and the bin loop has no bounds checks (iterate via
   `chunks_exact`, `get_unchecked` only where proven).
2. Bin count sweep K ∈ {8, 16, 32} × leaf-size sweep — pick the knee of the
   quality/speed curve (literature and practice say 16 is almost always it).
3. Allocation hygiene: single arena for indices, no per-node `Vec`s, `with_capacity`
   everywhere, reuse scratch buffers across the parallel tasks via `thread_local` or
   per-task scratch passed down.
4. Profile (`cargo flamegraph` / Instruments on macOS) — expected hotspots are the
   binning pass and the partition; both are memory-bound, so the SoA layout from
   Phase 0 is doing the heavy lifting already.

## Phase 6 (future / separate effort) — wide compressed BVH (CWBVH)

The actual state of the art for GPU traversal is an 8-wide compressed BVH
(Ylitie et al. 2017): 80-byte nodes, quantised child AABBs, octant-ordered traversal —
typically 1.5–2× faster than binary BVH on GPU. It requires a **shader rewrite** and a
new node format, so it's explicitly out of scope for this builder, but the plan above
feeds it directly: CWBVH is built by collapsing a high-quality binary SBVH — exactly
what Phases 1–3 produce. Keep the binary tree builder's output available as an
intermediate representation (not just the flattened `Vec<BvhNode>`) so a collapse pass
can be added later without touching the builder.

## Validation & acceptance criteria

| Metric | How | Target vs current `Bvh` |
|---|---|---|
| Build time (Sponza) | `benches/bvh.rs` side-by-side | ≥ 50× faster |
| SAH cost | Phase 0 metric printout | ≤ 0.9× (object splits), ≤ 0.75× (SBVH) |
| GPU traversal | `mesh_tests` counter / frame time on Sponza | fewer tests, lower frame time |
| Correctness | hit-parity test vs old BVH + invariant tests | exact nearest-hit parity (object splits) |

## Suggested implementation order

1. Phase 0 + Phase 1 together (one PR): correct fast sequential binned SAH + tests + metrics.
2. Phase 2 (parallelism) — small, isolated diff.
3. Phase 4.1/4.3 (DFS reorder, depth assert) — cheap wins.
4. Phase 3 (SBVH) behind a builder option.
5. Phase 5 tuning sweep, lock in constants.
