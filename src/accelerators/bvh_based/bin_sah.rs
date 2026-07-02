// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
// fable test
//! Hyper-optimised top-down binned-SAH BVH builder (Wald 2007), parallelised
//! with rayon, with opt-in spatial splits (Stich et al. 2009, "SBVH").
//!
//! Output follows the GPU traversal contract of `bvh.rs` / `intersects.comp`:
//! 32-byte nodes, root at index 0, sibling pairs adjacent at `idx` / `idx + 1`,
//! `amt != 0` marks a leaf covering triangles `[idx, idx + amt)` of the
//! reordered flat triangle buffer. Nodes are emitted depth-first so children
//! sit near their parents in the SSBO, and the final layout is deterministic
//! regardless of thread count.
//!
//! Build pipeline:
//! 1. SoA pre-pass: per-primitive AABB min/max into tight `Vec3A` arrays
//!    (centroids are recomputed as `(min + max) * 0.5` on the fly — the loads
//!    are needed for bin bounds anyway, so a third array would only cost
//!    bandwidth).
//! 2. Top-down build over an index (or reference) array into a bump-allocated
//!    intermediate node arena; per node a single pass bins all 3 axes into
//!    `BIN_COUNT` bins, a suffix/prefix sweep picks the SAH-optimal plane, and
//!    an in-place two-pointer partition splits the range. Large nodes bin
//!    chunk-parallel and recurse as rayon `join` tasks.
//! 3. DFS flatten of the arena into `Vec<BvhNode>`; leaf reference lists are
//!    emitted in DFS order, which manufactures the contiguous flat-triangle
//!    ranges the shader requires (and transparently handles SBVH duplication).
//! 4. Triangles are permuted (and, with spatial splits, duplicated) exactly
//!    once at the end.

use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

use glam::{Vec3, Vec3A};
use rayon::prelude::*;

use super::bvh::BvhNode;
use crate::accelerators::Accelerator;
use crate::{Material, Triangle};

/// SAH bins per axis; 16 is the knee of the quality/speed curve (Wald 2007).
const BIN_COUNT: usize = 16;
/// Must match `MAX_BVH_DEPTH` in `main.comp` (shader stack is `MAX_BVH_DEPTH + 1`).
const BVH_MAX_DEPTH: u32 = 64;
/// Ranges at least this large build their children as parallel rayon tasks.
const TASK_PARALLEL_MIN: usize = 8 * 1024;
/// Ranges at least this large bin chunk-parallel with per-chunk bin sets.
const CHUNK_PARALLEL_MIN: usize = 32 * 1024;
/// Chunk size for parallel binning.
const PAR_CHUNK: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SplitPolicy {
    /// Binned object splits only (fastest build, exact hit parity with any
    /// correct BVH over the same primitives).
    Object,
    /// Object splits plus spatial splits where the object-split children
    /// overlap significantly. `alpha` is the overlap-area/root-area threshold
    /// gating spatial-split evaluation; `budget` caps duplicated references as
    /// a fraction of the primitive count.
    Spatial { alpha: f32, budget: f32 },
}

impl SplitPolicy {
    pub fn spatial() -> Self {
        SplitPolicy::Spatial {
            alpha: 1e-5,
            budget: 0.3,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BinSah {
    pub policy: SplitPolicy,
    pub traversal_cost: f32,
    pub intersect_cost: f32,
    /// Ranges of at most this many references always become leaves.
    pub min_leaf: u32,
    /// Ranges larger than this are split even when the SAH prefers a leaf
    /// (median fallback), bounding leaf size.
    pub max_leaf: u32,
}

impl Default for BinSah {
    fn default() -> Self {
        Self {
            policy: SplitPolicy::Object,
            traversal_cost: 1.0,
            intersect_cost: 1.0,
            min_leaf: 2,
            max_leaf: 8,
        }
    }
}

impl BinSah {
    /// Final-quality preset: spatial splits with the standard SBVH budget.
    pub fn spatial() -> Self {
        Self {
            policy: SplitPolicy::spatial(),
            ..Self::default()
        }
    }

    pub fn build_with_stats(&self, triangles: &mut Vec<Triangle>) -> (Vec<BvhNode>, BuildStats) {
        let n = triangles.len();
        if n == 0 {
            // Inverted bounds never intersect, so traversal terminates at the root.
            let node = BvhNode::interior(Vec3::splat(1e30), Vec3::splat(-1e30), 0);
            return (vec![node], BuildStats::default());
        }
        assert!(n < u32::MAX as usize, "too many triangles for u32 indexing");

        let prims = Prims::build(triangles);

        let (arena, refs) = match self.policy {
            SplitPolicy::Object => {
                let mut idxs: Vec<u32> = (0..n as u32).collect();
                let arena = build_object_tree(self, &prims, &mut idxs);
                (arena, idxs)
            }
            SplitPolicy::Spatial { alpha, budget } => {
                build_spatial_tree(self, &prims, triangles, alpha, budget)
            }
        };
        drop(prims);

        let (nodes, order, mut stats) = flatten(self, &arena, &refs);
        stats.duplicated_refs = order.len().saturating_sub(n) as u32;
        apply_ordering(triangles, &order);
        (nodes, stats)
    }
}

impl Accelerator for BinSah {
    fn build(&self, triangles: &mut Vec<Triangle>, _materials: &mut Vec<Material>) -> Vec<BvhNode> {
        let (nodes, stats) = self.build_with_stats(triangles);
        log::debug!("bin_sah build: {stats:?}");
        nodes
    }
}

/// Post-build tree quality metrics.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuildStats {
    /// SAH cost of the tree, normalised by the root surface area.
    pub sah_cost: f32,
    pub node_count: u32,
    pub leaf_count: u32,
    pub max_depth: u32,
    pub max_leaf_size: u32,
    pub avg_leaf_size: f32,
    /// References beyond the input primitive count (spatial splits only).
    pub duplicated_refs: u32,
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Bounds {
    min: Vec3A,
    max: Vec3A,
}

impl Bounds {
    const EMPTY: Self = Self {
        min: Vec3A::INFINITY,
        max: Vec3A::NEG_INFINITY,
    };

    #[inline(always)]
    fn grow_point(&mut self, p: Vec3A) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    #[inline(always)]
    fn grow_minmax(&mut self, mn: Vec3A, mx: Vec3A) {
        self.min = self.min.min(mn);
        self.max = self.max.max(mx);
    }

    #[inline(always)]
    fn half_area(&self) -> f32 {
        // Clamping the diagonal makes empty/inverted bounds report zero area.
        let d = (self.max - self.min).max(Vec3A::ZERO);
        d.x * d.y + d.y * d.z + d.z * d.x
    }

    #[inline(always)]
    fn intersection(a: &Self, b: &Self) -> Self {
        Self {
            min: a.min.max(b.min),
            max: a.max.min(b.max),
        }
    }
}

#[inline(always)]
fn largest_axis(v: Vec3A) -> usize {
    if v.x >= v.y && v.x >= v.z {
        0
    } else if v.y >= v.z {
        1
    } else {
        2
    }
}

/// Per-axis bin scale; zero on degenerate axes so every centroid lands in bin 0.
#[inline(always)]
fn bin_scale(c: &Bounds) -> Vec3A {
    let ext = (c.max - c.min).max(Vec3A::ZERO);
    let k = Vec3A::splat(BIN_COUNT as f32 * (1.0 - 1e-6));
    Vec3A::select(ext.cmpgt(Vec3A::splat(1e-12)), k / ext, Vec3A::ZERO)
}

/// Bin index of a centroid on all three axes, clamped to `[0, BIN_COUNT)`.
/// Used by both binning and partitioning so the two always agree.
#[inline(always)]
fn bin_triple(cen: Vec3A, cb_min: Vec3A, scale: Vec3A) -> [usize; 3] {
    let t = ((cen - cb_min) * scale).clamp(Vec3A::ZERO, Vec3A::splat((BIN_COUNT - 1) as f32));
    [t.x as usize, t.y as usize, t.z as usize]
}

// ---------------------------------------------------------------------------
// SoA primitive data
// ---------------------------------------------------------------------------

struct Prims {
    min: Vec<Vec3A>,
    max: Vec<Vec3A>,
}

impl Prims {
    fn build(triangles: &[Triangle]) -> Self {
        let (min, max) = triangles
            .par_iter()
            .map(|t| (Vec3A::from(t.min_bound()), Vec3A::from(t.max_bound())))
            .unzip();
        Self { min, max }
    }

    fn len(&self) -> usize {
        self.min.len()
    }

    #[inline(always)]
    fn bounds(&self, i: u32) -> (Vec3A, Vec3A) {
        debug_assert!((i as usize) < self.min.len());
        // SAFETY: `i` comes from index arrays constructed over `0..len`.
        unsafe {
            (
                *self.min.get_unchecked(i as usize),
                *self.max.get_unchecked(i as usize),
            )
        }
    }

    /// Geometric and centroid bounds over all primitives.
    fn root_bounds(&self) -> (Bounds, Bounds) {
        self.min
            .par_iter()
            .zip(self.max.par_iter())
            .fold(
                || (Bounds::EMPTY, Bounds::EMPTY),
                |(mut g, mut c), (&mn, &mx)| {
                    g.grow_minmax(mn, mx);
                    c.grow_point((mn + mx) * 0.5);
                    (g, c)
                },
            )
            .reduce(
                || (Bounds::EMPTY, Bounds::EMPTY),
                |(mut ga, mut ca), (gb, cb)| {
                    ga.grow_minmax(gb.min, gb.max);
                    ca.grow_minmax(cb.min, cb.max);
                    (ga, ca)
                },
            )
    }
}

// ---------------------------------------------------------------------------
// Binning (object splits)
// ---------------------------------------------------------------------------

/// 3 axes x BIN_COUNT bins of geometric bounds + counts, SoA so the reset is a
/// handful of memsets and the hot loop stays branch-free.
#[derive(Clone, Copy)]
struct Bins {
    gmin: [[Vec3A; BIN_COUNT]; 3],
    gmax: [[Vec3A; BIN_COUNT]; 3],
    count: [[u32; BIN_COUNT]; 3],
}

impl Bins {
    fn new() -> Self {
        Self {
            gmin: [[Vec3A::INFINITY; BIN_COUNT]; 3],
            gmax: [[Vec3A::NEG_INFINITY; BIN_COUNT]; 3],
            count: [[0; BIN_COUNT]; 3],
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    #[inline(always)]
    fn add(&mut self, t: [usize; 3], mn: Vec3A, mx: Vec3A) {
        for (axis, &b) in t.iter().enumerate() {
            debug_assert!(b < BIN_COUNT);
            // SAFETY: `bin_triple` clamps every component to [0, BIN_COUNT).
            unsafe {
                let gm = self.gmin.get_unchecked_mut(axis).get_unchecked_mut(b);
                *gm = gm.min(mn);
                let gx = self.gmax.get_unchecked_mut(axis).get_unchecked_mut(b);
                *gx = gx.max(mx);
                *self.count.get_unchecked_mut(axis).get_unchecked_mut(b) += 1;
            }
        }
    }

    fn merge(mut self, o: Self) -> Self {
        for axis in 0..3 {
            for b in 0..BIN_COUNT {
                self.gmin[axis][b] = self.gmin[axis][b].min(o.gmin[axis][b]);
                self.gmax[axis][b] = self.gmax[axis][b].max(o.gmax[axis][b]);
                self.count[axis][b] += o.count[axis][b];
            }
        }
        self
    }
}

#[inline]
fn bin_iter(
    it: impl Iterator<Item = (Vec3A, Vec3A)>,
    cb_min: Vec3A,
    scale: Vec3A,
    bins: &mut Bins,
) {
    for (mn, mx) in it {
        let cen = (mn + mx) * 0.5;
        bins.add(bin_triple(cen, cb_min, scale), mn, mx);
    }
}

fn bin_prims(prims: &Prims, idxs: &[u32], cb_min: Vec3A, scale: Vec3A, bins: &mut Bins) {
    if idxs.len() >= CHUNK_PARALLEL_MIN {
        *bins = idxs
            .par_chunks(PAR_CHUNK)
            .map(|chunk| {
                let mut local = Bins::new();
                bin_iter(
                    chunk.iter().map(|&i| prims.bounds(i)),
                    cb_min,
                    scale,
                    &mut local,
                );
                local
            })
            .reduce(Bins::new, Bins::merge);
    } else {
        bins.reset();
        bin_iter(idxs.iter().map(|&i| prims.bounds(i)), cb_min, scale, bins);
    }
}

/// Best SAH split over all axes: `(unnormalised cost, axis, split bin)` where
/// bins `[0, split)` go left. `None` when no plane separates the centroids.
fn evaluate_sah(bins: &Bins) -> Option<(f32, usize, usize)> {
    let mut best_cost = f32::INFINITY;
    let mut best_axis = usize::MAX;
    let mut best_bin = 0usize;
    for axis in 0..3 {
        let mut right_h = [0.0f32; BIN_COUNT];
        let mut right_n = [0u32; BIN_COUNT];
        let mut acc = Bounds::EMPTY;
        let mut cnt = 0u32;
        for i in (1..BIN_COUNT).rev() {
            acc.grow_minmax(bins.gmin[axis][i], bins.gmax[axis][i]);
            cnt += bins.count[axis][i];
            right_h[i] = acc.half_area();
            right_n[i] = cnt;
        }
        let mut acc = Bounds::EMPTY;
        let mut cnt = 0u32;
        for i in 1..BIN_COUNT {
            acc.grow_minmax(bins.gmin[axis][i - 1], bins.gmax[axis][i - 1]);
            cnt += bins.count[axis][i - 1];
            if cnt == 0 || right_n[i] == 0 {
                continue;
            }
            let cost = cnt as f32 * acc.half_area() + right_n[i] as f32 * right_h[i];
            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_bin = i;
            }
        }
    }
    (best_axis != usize::MAX).then_some((best_cost, best_axis, best_bin))
}

fn union_bins(bins: &Bins, axis: usize, lo: usize, hi: usize) -> (Bounds, u32) {
    let mut g = Bounds::EMPTY;
    let mut n = 0u32;
    for b in lo..hi {
        g.grow_minmax(bins.gmin[axis][b], bins.gmax[axis][b]);
        n += bins.count[axis][b];
    }
    (g, n)
}

// ---------------------------------------------------------------------------
// Intermediate tree arena
// ---------------------------------------------------------------------------

/// Intermediate binary tree node; flattened to `BvhNode` after the build.
/// `b != 0`: leaf over references `[a, a + b)`; `b == 0`: interior with
/// children at arena slots `a` and `a + 1`.
#[derive(Clone, Copy)]
struct BNode {
    min: Vec3A,
    max: Vec3A,
    a: u32,
    b: u32,
}

impl BNode {
    #[inline]
    fn new_leaf(g: &Bounds, first: u32, count: u32) -> Self {
        debug_assert!(count != 0);
        Self {
            min: g.min,
            max: g.max,
            a: first,
            b: count,
        }
    }

    #[inline]
    fn new_interior(g: &Bounds, left: u32) -> Self {
        Self {
            min: g.min,
            max: g.max,
            a: left,
            b: 0,
        }
    }
}

/// Shared bump allocator over a pre-reserved `Vec<BNode>`. `fetch_add(2)`
/// hands every task disjoint sibling-adjacent slots, so writes never alias and
/// no fixup pass is needed; the DFS flatten normalises placement afterwards,
/// which keeps the final output deterministic regardless of scheduling.
struct NodeArena {
    ptr: *mut BNode,
    cap: usize,
    cursor: AtomicUsize,
}

// SAFETY: every slot is written by exactly one task (slots are handed out by
// `fetch_add`), and the consumer only reads after all tasks joined.
unsafe impl Send for NodeArena {}
unsafe impl Sync for NodeArena {}

impl NodeArena {
    fn new(ptr: *mut BNode, cap: usize) -> Self {
        Self {
            ptr,
            cap,
            // Slot 0 is the root.
            cursor: AtomicUsize::new(1),
        }
    }

    #[inline]
    fn alloc_pair(&self) -> u32 {
        let i = self.cursor.fetch_add(2, Ordering::Relaxed);
        assert!(i + 2 <= self.cap, "bvh node arena overflow");
        i as u32
    }

    #[inline]
    unsafe fn write(&self, slot: u32, node: BNode) {
        debug_assert!((slot as usize) < self.cap);
        // SAFETY: slot is within the reserved capacity and owned by the caller.
        unsafe { self.ptr.add(slot as usize).write(node) };
    }

    fn used(&self) -> usize {
        self.cursor.load(Ordering::Relaxed)
    }
}

/// Shared bump pool for leaf primitive ids (spatial-split builds).
struct RefPool {
    ptr: *mut u32,
    cap: usize,
    cursor: AtomicUsize,
}

// SAFETY: disjoint ranges are handed out by `fetch_add`; reads happen after join.
unsafe impl Send for RefPool {}
unsafe impl Sync for RefPool {}

impl RefPool {
    fn new(ptr: *mut u32, cap: usize) -> Self {
        Self {
            ptr,
            cap,
            cursor: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn alloc(&self, n: usize) -> usize {
        let at = self.cursor.fetch_add(n, Ordering::Relaxed);
        assert!(at + n <= self.cap, "bvh reference pool overflow");
        at
    }

    #[inline]
    unsafe fn write(&self, at: usize, vals: &[u32]) {
        // SAFETY: `[at, at + vals.len())` was reserved by `alloc` for this caller.
        unsafe { std::ptr::copy_nonoverlapping(vals.as_ptr(), self.ptr.add(at), vals.len()) };
    }

    fn used(&self) -> usize {
        self.cursor.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Object-split builder
// ---------------------------------------------------------------------------

struct ObjCtx<'a> {
    cfg: BinSah,
    prims: &'a Prims,
    arena: &'a NodeArena,
}

enum Decision {
    Leaf,
    Split {
        mid: usize,
        lg: Bounds,
        lc: Bounds,
        rg: Bounds,
        rc: Bounds,
    },
}

fn build_object_tree(cfg: &BinSah, prims: &Prims, idxs: &mut [u32]) -> Vec<BNode> {
    let n = idxs.len();
    let (g, c) = prims.root_bounds();
    // A binary tree over n leaves of >= 1 primitive has at most 2n - 1 nodes.
    let cap = 2 * n;
    let mut storage: Vec<BNode> = Vec::with_capacity(cap);
    let arena = NodeArena::new(storage.as_mut_ptr(), cap);
    {
        let ctx = ObjCtx {
            cfg: *cfg,
            prims,
            arena: &arena,
        };
        build_par(&ctx, 0, idxs, 0, g, c, 0);
    }
    let used = arena.used();
    // SAFETY: every slot in [0, used) was written exactly once before the
    // tasks joined.
    unsafe { storage.set_len(used) };
    storage
}

fn build_par(
    ctx: &ObjCtx,
    slot: u32,
    idxs: &mut [u32],
    base: u32,
    g: Bounds,
    c: Bounds,
    depth: u32,
) {
    if idxs.len() < TASK_PARALLEL_MIN {
        build_seq(ctx, slot, idxs, base, g, c, depth);
        return;
    }
    let mut bins = Bins::new();
    match decide_split(&ctx.cfg, ctx.prims, idxs, &g, &c, depth, &mut bins) {
        Decision::Leaf => {
            // SAFETY: `slot` is owned by this task.
            unsafe {
                ctx.arena
                    .write(slot, BNode::new_leaf(&g, base, idxs.len() as u32))
            };
        }
        Decision::Split {
            mid,
            lg,
            lc,
            rg,
            rc,
        } => {
            let pair = ctx.arena.alloc_pair();
            // SAFETY: `slot` is owned by this task.
            unsafe { ctx.arena.write(slot, BNode::new_interior(&g, pair)) };
            let (l, r) = idxs.split_at_mut(mid);
            rayon::join(
                || build_par(ctx, pair, l, base, lg, lc, depth + 1),
                || build_par(ctx, pair + 1, r, base + mid as u32, rg, rc, depth + 1),
            );
        }
    }
}

struct SeqItem {
    slot: u32,
    lo: u32,
    hi: u32,
    g: Bounds,
    c: Bounds,
    depth: u32,
}

/// Iterative sequential build with an explicit work stack and a single reused
/// bin set — no recursion, no per-node allocations.
fn build_seq(
    ctx: &ObjCtx,
    slot: u32,
    idxs: &mut [u32],
    base: u32,
    g: Bounds,
    c: Bounds,
    depth: u32,
) {
    let mut bins = Bins::new();
    let mut stack: Vec<SeqItem> = Vec::with_capacity(64);
    stack.push(SeqItem {
        slot,
        lo: 0,
        hi: idxs.len() as u32,
        g,
        c,
        depth,
    });
    while let Some(it) = stack.pop() {
        let range = &mut idxs[it.lo as usize..it.hi as usize];
        match decide_split(
            &ctx.cfg, ctx.prims, range, &it.g, &it.c, it.depth, &mut bins,
        ) {
            Decision::Leaf => {
                // SAFETY: `it.slot` is owned by this task.
                unsafe {
                    ctx.arena
                        .write(it.slot, BNode::new_leaf(&it.g, base + it.lo, it.hi - it.lo))
                };
            }
            Decision::Split {
                mid,
                lg,
                lc,
                rg,
                rc,
            } => {
                let pair = ctx.arena.alloc_pair();
                // SAFETY: `it.slot` is owned by this task.
                unsafe { ctx.arena.write(it.slot, BNode::new_interior(&it.g, pair)) };
                let m = it.lo + mid as u32;
                stack.push(SeqItem {
                    slot: pair + 1,
                    lo: m,
                    hi: it.hi,
                    g: rg,
                    c: rc,
                    depth: it.depth + 1,
                });
                stack.push(SeqItem {
                    slot: pair,
                    lo: it.lo,
                    hi: m,
                    g: lg,
                    c: lc,
                    depth: it.depth + 1,
                });
            }
        }
    }
}

fn decide_split(
    cfg: &BinSah,
    prims: &Prims,
    idxs: &mut [u32],
    g: &Bounds,
    c: &Bounds,
    depth: u32,
    bins: &mut Bins,
) -> Decision {
    let n = idxs.len();
    if n <= cfg.min_leaf as usize || depth >= BVH_MAX_DEPTH {
        return Decision::Leaf;
    }

    let scale = bin_scale(c);
    if !scale.cmpgt(Vec3A::ZERO).any() {
        // All centroids coincide: no plane can separate them.
        return if n <= cfg.max_leaf as usize {
            Decision::Leaf
        } else {
            median_split(prims, idxs, c, g)
        };
    }

    bin_prims(prims, idxs, c.min, scale, bins);
    let Some((best_unnorm, axis, bin)) = evaluate_sah(bins) else {
        return if n <= cfg.max_leaf as usize {
            Decision::Leaf
        } else {
            median_split(prims, idxs, c, g)
        };
    };

    let parent_h = g.half_area().max(1e-30);
    let split_cost = cfg.traversal_cost + cfg.intersect_cost * best_unnorm / parent_h;
    if split_cost >= n as f32 * cfg.intersect_cost {
        return if n <= cfg.max_leaf as usize {
            Decision::Leaf
        } else {
            median_split(prims, idxs, c, g)
        };
    }

    let (mid, lc, rc) = partition_by_bin(prims, idxs, axis, bin, c.min, scale);
    if mid == 0 || mid == n {
        // Numeric safety net; bin counts guarantee both sides are non-empty.
        return median_split(prims, idxs, c, g);
    }
    let (lg, ln) = union_bins(bins, axis, 0, bin);
    let (rg, _) = union_bins(bins, axis, bin, BIN_COUNT);
    debug_assert_eq!(mid, ln as usize);
    Decision::Split {
        mid,
        lg,
        lc,
        rg,
        rc,
    }
}

/// In-place two-pointer partition by bin index (`< split` goes left); each
/// element is classified exactly once and the child centroid bounds are
/// accumulated on the way through.
fn partition_by_bin(
    prims: &Prims,
    idxs: &mut [u32],
    axis: usize,
    split: usize,
    cb_min: Vec3A,
    scale: Vec3A,
) -> (usize, Bounds, Bounds) {
    let mut lc = Bounds::EMPTY;
    let mut rc = Bounds::EMPTY;
    let mut l = 0usize;
    let mut r = idxs.len();
    while l < r {
        let (mn, mx) = prims.bounds(idxs[l]);
        let cen = (mn + mx) * 0.5;
        if bin_triple(cen, cb_min, scale)[axis] < split {
            lc.grow_point(cen);
            l += 1;
        } else {
            rc.grow_point(cen);
            r -= 1;
            idxs.swap(l, r);
        }
    }
    (l, lc, rc)
}

#[inline(always)]
fn centroid_axis(prims: &Prims, i: u32, axis: usize) -> f32 {
    let (mn, mx) = prims.bounds(i);
    (mn[axis] + mx[axis]) * 0.5
}

fn bounds_of(prims: &Prims, idxs: &[u32]) -> (Bounds, Bounds) {
    let mut g = Bounds::EMPTY;
    let mut c = Bounds::EMPTY;
    for &i in idxs {
        let (mn, mx) = prims.bounds(i);
        g.grow_minmax(mn, mx);
        c.grow_point((mn + mx) * 0.5);
    }
    (g, c)
}

/// Forced even split at the centroid median of the widest axis; used when the
/// SAH refuses to split a range that exceeds `max_leaf` or when the centroid
/// bounds are degenerate.
fn median_split(prims: &Prims, idxs: &mut [u32], c: &Bounds, g: &Bounds) -> Decision {
    let n = idxs.len();
    let mid = n / 2;
    let cext = (c.max - c.min).max(Vec3A::ZERO);
    let axis = if cext.max_element() > 0.0 {
        largest_axis(cext)
    } else {
        largest_axis((g.max - g.min).max(Vec3A::ZERO))
    };
    idxs.select_nth_unstable_by(mid, |&p, &q| {
        centroid_axis(prims, p, axis).total_cmp(&centroid_axis(prims, q, axis))
    });
    let (lg, lc) = bounds_of(prims, &idxs[..mid]);
    let (rg, rc) = bounds_of(prims, &idxs[mid..]);
    Decision::Split {
        mid,
        lg,
        lc,
        rg,
        rc,
    }
}

// ---------------------------------------------------------------------------
// Spatial-split builder (SBVH)
// ---------------------------------------------------------------------------

/// A build reference: a primitive with a (possibly clipped) bounding box.
#[derive(Clone, Copy)]
struct SRef {
    min: Vec3A,
    max: Vec3A,
    prim: u32,
}

type Side = (Vec<SRef>, Bounds, Bounds);

struct SCtx<'a> {
    cfg: BinSah,
    alpha: f32,
    tris: &'a [Triangle],
    arena: &'a NodeArena,
    pool: &'a RefPool,
    budget: &'a AtomicIsize,
    root_half: f32,
}

struct ObjCand {
    cost: f32,
    axis: usize,
    bin: usize,
    lg: Bounds,
    rg: Bounds,
}

struct SpatCand {
    cost: f32,
    axis: usize,
    plane: f32,
}

fn build_spatial_tree(
    cfg: &BinSah,
    prims: &Prims,
    tris: &[Triangle],
    alpha: f32,
    budget_frac: f32,
) -> (Vec<BNode>, Vec<u32>) {
    let n = prims.len();
    let (g, c) = prims.root_bounds();

    let extra = (n as f64 * budget_frac.max(0.0) as f64) as usize;
    let pool_cap = n + extra;
    let node_cap = 2 * pool_cap;

    let mut node_storage: Vec<BNode> = Vec::with_capacity(node_cap);
    let mut pool_storage: Vec<u32> = Vec::with_capacity(pool_cap);
    let arena = NodeArena::new(node_storage.as_mut_ptr(), node_cap);
    let pool = RefPool::new(pool_storage.as_mut_ptr(), pool_cap);
    let budget = AtomicIsize::new(extra as isize);

    let refs: Vec<SRef> = (0..n as u32)
        .into_par_iter()
        .map(|i| {
            let (mn, mx) = prims.bounds(i);
            SRef {
                min: mn,
                max: mx,
                prim: i,
            }
        })
        .collect();

    {
        let ctx = SCtx {
            cfg: *cfg,
            alpha,
            tris,
            arena: &arena,
            pool: &pool,
            budget: &budget,
            root_half: g.half_area().max(1e-30),
        };
        build_sbvh(&ctx, 0, refs, g, c, 0);
    }

    // SAFETY: all allocated slots/ranges were written before the tasks joined.
    unsafe {
        node_storage.set_len(arena.used());
        pool_storage.set_len(pool.used());
    }
    (node_storage, pool_storage)
}

fn build_sbvh(ctx: &SCtx, slot: u32, mut refs: Vec<SRef>, g: Bounds, c: Bounds, depth: u32) {
    let cfg = &ctx.cfg;
    let n = refs.len();
    if n <= cfg.min_leaf as usize || depth >= BVH_MAX_DEPTH {
        emit_sbvh_leaf(ctx, slot, &refs, &g);
        return;
    }

    // Object-split candidate (binned SAH over reference centroids).
    let scale = bin_scale(&c);
    let mut object: Option<ObjCand> = None;
    if scale.cmpgt(Vec3A::ZERO).any() {
        let mut bins = Bins::new();
        bin_srefs(&refs, c.min, scale, &mut bins);
        if let Some((cost, axis, bin)) = evaluate_sah(&bins) {
            let (lg, _) = union_bins(&bins, axis, 0, bin);
            let (rg, _) = union_bins(&bins, axis, bin, BIN_COUNT);
            object = Some(ObjCand {
                cost,
                axis,
                bin,
                lg,
                rg,
            });
        }
    }

    // Spatial-split candidate, gated on child overlap (Stich's lambda > alpha)
    // so most nodes never pay for chopped binning.
    let try_spatial = ctx.budget.load(Ordering::Relaxed) > 0
        && match &object {
            Some(o) => Bounds::intersection(&o.lg, &o.rg).half_area() / ctx.root_half > ctx.alpha,
            None => true,
        };
    let spatial = if try_spatial {
        eval_spatial(ctx, &refs, &g)
    } else {
        None
    };

    let parent_h = g.half_area().max(1e-30);
    let norm = |unnorm: f32| cfg.traversal_cost + cfg.intersect_cost * unnorm / parent_h;
    let obj_cost = object.as_ref().map(|o| norm(o.cost));
    let spa_cost = spatial.as_ref().map(|s| norm(s.cost));

    let prefer_spatial = match (obj_cost, spa_cost) {
        (Some(o), Some(s)) => s < o,
        (None, Some(_)) => true,
        _ => false,
    };
    let best_cost = match (obj_cost, spa_cost) {
        (Some(o), Some(s)) => Some(o.min(s)),
        (Some(o), None) => Some(o),
        (None, Some(s)) => Some(s),
        (None, None) => None,
    };

    if !best_cost.is_some_and(|bc| bc < n as f32 * cfg.intersect_cost) {
        if n <= cfg.max_leaf as usize {
            emit_sbvh_leaf(ctx, slot, &refs, &g);
            return;
        }
        let (l, r) = median_split_refs(refs, &g, &c);
        sbvh_children(ctx, slot, &g, l, r, depth);
        return;
    }

    if prefer_spatial {
        let s = spatial.as_ref().unwrap();
        if let Some((l, r)) = distribute_spatial(ctx, &refs, s.axis, s.plane) {
            drop(refs);
            sbvh_children(ctx, slot, &g, l, r, depth);
            return;
        }
        // Degenerate spatial distribution; fall through to the object split.
    }

    if let Some(o) = object {
        let (mid, lc, rc) = partition_srefs(&mut refs, o.axis, o.bin, c.min, scale);
        if mid > 0 && mid < refs.len() {
            let right = refs.split_off(mid);
            sbvh_children(ctx, slot, &g, (refs, o.lg, lc), (right, o.rg, rc), depth);
            return;
        }
    }

    let (l, r) = median_split_refs(refs, &g, &c);
    sbvh_children(ctx, slot, &g, l, r, depth);
}

fn sbvh_children(ctx: &SCtx, slot: u32, g: &Bounds, left: Side, right: Side, depth: u32) {
    let pair = ctx.arena.alloc_pair();
    // SAFETY: `slot` is owned by this task.
    unsafe { ctx.arena.write(slot, BNode::new_interior(g, pair)) };
    let (lrefs, lg, lc) = left;
    let (rrefs, rg, rc) = right;
    let d = depth + 1;
    if lrefs.len().max(rrefs.len()) >= TASK_PARALLEL_MIN {
        rayon::join(
            || build_sbvh(ctx, pair, lrefs, lg, lc, d),
            || build_sbvh(ctx, pair + 1, rrefs, rg, rc, d),
        );
    } else {
        build_sbvh(ctx, pair, lrefs, lg, lc, d);
        build_sbvh(ctx, pair + 1, rrefs, rg, rc, d);
    }
}

fn emit_sbvh_leaf(ctx: &SCtx, slot: u32, refs: &[SRef], g: &Bounds) {
    // Spatial splits can land both halves of a primitive in the same leaf;
    // dedupe so the GPU does not test the triangle twice.
    let mut ids: Vec<u32> = refs.iter().map(|r| r.prim).collect();
    ids.sort_unstable();
    ids.dedup();
    let at = ctx.pool.alloc(ids.len());
    // SAFETY: the range was just reserved; the slot is owned by this task.
    unsafe {
        ctx.pool.write(at, &ids);
        ctx.arena
            .write(slot, BNode::new_leaf(g, at as u32, ids.len() as u32));
    }
}

fn bin_srefs(refs: &[SRef], cb_min: Vec3A, scale: Vec3A, bins: &mut Bins) {
    if refs.len() >= CHUNK_PARALLEL_MIN {
        *bins = refs
            .par_chunks(PAR_CHUNK)
            .map(|chunk| {
                let mut local = Bins::new();
                bin_iter(
                    chunk.iter().map(|r| (r.min, r.max)),
                    cb_min,
                    scale,
                    &mut local,
                );
                local
            })
            .reduce(Bins::new, Bins::merge);
    } else {
        bin_iter(refs.iter().map(|r| (r.min, r.max)), cb_min, scale, bins);
    }
}

fn partition_srefs(
    refs: &mut [SRef],
    axis: usize,
    split: usize,
    cb_min: Vec3A,
    scale: Vec3A,
) -> (usize, Bounds, Bounds) {
    let mut lc = Bounds::EMPTY;
    let mut rc = Bounds::EMPTY;
    let mut l = 0usize;
    let mut r = refs.len();
    while l < r {
        let cen = (refs[l].min + refs[l].max) * 0.5;
        if bin_triple(cen, cb_min, scale)[axis] < split {
            lc.grow_point(cen);
            l += 1;
        } else {
            rc.grow_point(cen);
            r -= 1;
            refs.swap(l, r);
        }
    }
    (l, lc, rc)
}

fn bounds_of_refs(refs: &[SRef]) -> (Bounds, Bounds) {
    let mut g = Bounds::EMPTY;
    let mut c = Bounds::EMPTY;
    for r in refs {
        g.grow_minmax(r.min, r.max);
        c.grow_point((r.min + r.max) * 0.5);
    }
    (g, c)
}

fn median_split_refs(mut refs: Vec<SRef>, g: &Bounds, c: &Bounds) -> (Side, Side) {
    let n = refs.len();
    let mid = n / 2;
    let cext = (c.max - c.min).max(Vec3A::ZERO);
    let axis = if cext.max_element() > 0.0 {
        largest_axis(cext)
    } else {
        largest_axis((g.max - g.min).max(Vec3A::ZERO))
    };
    refs.select_nth_unstable_by(mid, |a, b| {
        ((a.min[axis] + a.max[axis]) * 0.5).total_cmp(&((b.min[axis] + b.max[axis]) * 0.5))
    });
    let right = refs.split_off(mid);
    let (lg, lc) = bounds_of_refs(&refs);
    let (rg, rc) = bounds_of_refs(&right);
    ((refs, lg, lc), (right, rg, rc))
}

// --- chopped binning -------------------------------------------------------

struct SpatialBins {
    gmin: [Vec3A; BIN_COUNT],
    gmax: [Vec3A; BIN_COUNT],
    entry: [u32; BIN_COUNT],
    exit: [u32; BIN_COUNT],
}

impl SpatialBins {
    fn new() -> Self {
        Self {
            gmin: [Vec3A::INFINITY; BIN_COUNT],
            gmax: [Vec3A::NEG_INFINITY; BIN_COUNT],
            entry: [0; BIN_COUNT],
            exit: [0; BIN_COUNT],
        }
    }

    #[inline(always)]
    fn grow(&mut self, b: usize, mn: Vec3A, mx: Vec3A) {
        self.gmin[b] = self.gmin[b].min(mn);
        self.gmax[b] = self.gmax[b].max(mx);
    }

    fn merge(mut self, o: Self) -> Self {
        for b in 0..BIN_COUNT {
            self.gmin[b] = self.gmin[b].min(o.gmin[b]);
            self.gmax[b] = self.gmax[b].max(o.gmax[b]);
            self.entry[b] += o.entry[b];
            self.exit[b] += o.exit[b];
        }
        self
    }
}

/// Convex polygon on the stack; a triangle chopped by parallel axis planes
/// never exceeds 5 vertices per piece, 8 leaves headroom for on-plane cases.
#[derive(Clone, Copy)]
struct Poly {
    v: [Vec3A; 8],
    n: u32,
}

impl Poly {
    fn from_triangle(tri: &Triangle) -> Self {
        let mut v = [Vec3A::ZERO; 8];
        v[0] = Vec3A::from(tri.vertices[0].0);
        v[1] = Vec3A::from(tri.vertices[1].0);
        v[2] = Vec3A::from(tri.vertices[2].0);
        Self { v, n: 3 }
    }

    #[inline]
    fn push(&mut self, p: Vec3A) {
        if (self.n as usize) < self.v.len() {
            self.v[self.n as usize] = p;
            self.n += 1;
        }
    }

    /// Sutherland–Hodgman split against `v[axis] = pos`: (part <= pos, part >= pos).
    fn split(&self, axis: usize, pos: f32) -> (Poly, Poly) {
        let empty = Poly {
            v: [Vec3A::ZERO; 8],
            n: 0,
        };
        let mut left = empty;
        let mut right = empty;
        if self.n == 0 {
            return (left, right);
        }
        for e in 0..self.n as usize {
            let p = self.v[e];
            let q = self.v[(e + 1) % self.n as usize];
            let dp = p[axis] - pos;
            let dq = q[axis] - pos;
            if dp <= 0.0 {
                left.push(p);
            }
            if dp >= 0.0 {
                right.push(p);
            }
            if (dp < 0.0 && dq > 0.0) || (dp > 0.0 && dq < 0.0) {
                let m = p + (q - p) * (dp / (dp - dq));
                left.push(m);
                right.push(m);
            }
        }
        (left, right)
    }

    fn bounds(&self) -> Option<(Vec3A, Vec3A)> {
        if self.n == 0 {
            return None;
        }
        let mut mn = self.v[0];
        let mut mx = self.v[0];
        for i in 1..self.n as usize {
            mn = mn.min(self.v[i]);
            mx = mx.max(self.v[i]);
        }
        Some((mn, mx))
    }
}

/// Polygon bounds intersected with the parent reference box (keeps clipped
/// reference boxes monotonically tightening down the tree).
#[inline]
fn clip_box(poly: &Poly, rmin: Vec3A, rmax: Vec3A) -> Option<(Vec3A, Vec3A)> {
    let (mn, mx) = poly.bounds()?;
    let mn = mn.max(rmin);
    let mx = mx.min(rmax);
    if mn.cmple(mx).all() {
        Some((mn, mx))
    } else {
        None
    }
}

fn spatial_bins_for(
    ctx: &SCtx,
    refs: &[SRef],
    axis: usize,
    base: f32,
    width: f32,
    inv: f32,
) -> SpatialBins {
    let top = BIN_COUNT - 1;
    let mut sb = SpatialBins::new();
    for r in refs {
        let lo = (((r.min[axis] - base) * inv).max(0.0) as usize).min(top);
        let hi = ((((r.max[axis] - base) * inv).max(0.0) as usize).min(top)).max(lo);
        if lo == hi {
            sb.grow(lo, r.min, r.max);
            sb.entry[lo] += 1;
            sb.exit[lo] += 1;
            continue;
        }
        let tri = &ctx.tris[r.prim as usize];
        if tri.is_sphere.0 == 1 {
            // Spheres are never spatially split: whole reference at its centroid bin.
            let cen = (r.min[axis] + r.max[axis]) * 0.5;
            let b = (((cen - base) * inv).max(0.0) as usize).min(top);
            sb.grow(b, r.min, r.max);
            sb.entry[b] += 1;
            sb.exit[b] += 1;
            continue;
        }
        // Chop the actual triangle polygon across the spanned bins.
        let mut poly = Poly::from_triangle(tri);
        for b in lo..hi {
            let plane = base + width * (b as f32 + 1.0);
            let (lp, rp) = poly.split(axis, plane);
            if let Some((mn, mx)) = clip_box(&lp, r.min, r.max) {
                sb.grow(b, mn, mx);
            }
            poly = rp;
            if poly.n == 0 {
                break;
            }
        }
        if let Some((mn, mx)) = clip_box(&poly, r.min, r.max) {
            sb.grow(hi, mn, mx);
        }
        sb.entry[lo] += 1;
        sb.exit[hi] += 1;
    }
    sb
}

fn eval_spatial(ctx: &SCtx, refs: &[SRef], g: &Bounds) -> Option<SpatCand> {
    let ext = (g.max - g.min).max(Vec3A::ZERO);
    let mut best: Option<SpatCand> = None;
    for axis in 0..3 {
        if ext[axis] <= 1e-12 {
            continue;
        }
        let width = ext[axis] / BIN_COUNT as f32;
        let inv = BIN_COUNT as f32 * (1.0 - 1e-6) / ext[axis];
        let base = g.min[axis];
        let sb = if refs.len() >= CHUNK_PARALLEL_MIN {
            refs.par_chunks(PAR_CHUNK)
                .map(|chunk| spatial_bins_for(ctx, chunk, axis, base, width, inv))
                .reduce(SpatialBins::new, SpatialBins::merge)
        } else {
            spatial_bins_for(ctx, refs, axis, base, width, inv)
        };

        let mut right_h = [0.0f32; BIN_COUNT];
        let mut right_n = [0u32; BIN_COUNT];
        let mut acc = Bounds::EMPTY;
        let mut cnt = 0u32;
        for i in (1..BIN_COUNT).rev() {
            acc.grow_minmax(sb.gmin[i], sb.gmax[i]);
            cnt += sb.exit[i];
            right_h[i] = acc.half_area();
            right_n[i] = cnt;
        }
        let mut acc = Bounds::EMPTY;
        let mut cnt = 0u32;
        for i in 1..BIN_COUNT {
            acc.grow_minmax(sb.gmin[i - 1], sb.gmax[i - 1]);
            cnt += sb.entry[i - 1];
            if cnt == 0 || right_n[i] == 0 {
                continue;
            }
            let cost = cnt as f32 * acc.half_area() + right_n[i] as f32 * right_h[i];
            if best.as_ref().is_none_or(|b| cost < b.cost) {
                best = Some(SpatCand {
                    cost,
                    axis,
                    plane: base + width * i as f32,
                });
            }
        }
    }
    best
}

#[inline]
fn try_take_budget(ctx: &SCtx) -> bool {
    if ctx.budget.fetch_sub(1, Ordering::Relaxed) > 0 {
        true
    } else {
        ctx.budget.fetch_add(1, Ordering::Relaxed);
        false
    }
}

#[inline]
fn refund_budget(ctx: &SCtx) {
    ctx.budget.fetch_add(1, Ordering::Relaxed);
}

fn distribute_spatial(ctx: &SCtx, refs: &[SRef], axis: usize, plane: f32) -> Option<(Side, Side)> {
    #[inline(always)]
    fn push(side: &mut Vec<SRef>, g: &mut Bounds, c: &mut Bounds, mn: Vec3A, mx: Vec3A, prim: u32) {
        g.grow_minmax(mn, mx);
        c.grow_point((mn + mx) * 0.5);
        side.push(SRef {
            min: mn,
            max: mx,
            prim,
        });
    }

    let mut l: Vec<SRef> = Vec::with_capacity(refs.len());
    let mut r: Vec<SRef> = Vec::with_capacity(refs.len());
    let (mut lg, mut lc) = (Bounds::EMPTY, Bounds::EMPTY);
    let (mut rg, mut rc) = (Bounds::EMPTY, Bounds::EMPTY);

    for rf in refs {
        if rf.max[axis] <= plane {
            push(&mut l, &mut lg, &mut lc, rf.min, rf.max, rf.prim);
        } else if rf.min[axis] >= plane {
            push(&mut r, &mut rg, &mut rc, rf.min, rf.max, rf.prim);
        } else {
            let tri = &ctx.tris[rf.prim as usize];
            let splittable = tri.is_sphere.0 != 1;
            if splittable && try_take_budget(ctx) {
                let poly = Poly::from_triangle(tri);
                let (lp, rp) = poly.split(axis, plane);
                match (clip_box(&lp, rf.min, rf.max), clip_box(&rp, rf.min, rf.max)) {
                    (Some((lmn, lmx)), Some((rmn, rmx))) => {
                        push(&mut l, &mut lg, &mut lc, lmn, lmx, rf.prim);
                        push(&mut r, &mut rg, &mut rc, rmn, rmx, rf.prim);
                    }
                    (Some((mn, mx)), None) => {
                        refund_budget(ctx);
                        push(&mut l, &mut lg, &mut lc, mn, mx, rf.prim);
                    }
                    (None, Some((mn, mx))) => {
                        refund_budget(ctx);
                        push(&mut r, &mut rg, &mut rc, mn, mx, rf.prim);
                    }
                    (None, None) => {
                        refund_budget(ctx);
                        let cen = (rf.min[axis] + rf.max[axis]) * 0.5;
                        if cen < plane {
                            push(&mut l, &mut lg, &mut lc, rf.min, rf.max, rf.prim);
                        } else {
                            push(&mut r, &mut rg, &mut rc, rf.min, rf.max, rf.prim);
                        }
                    }
                }
            } else {
                // Unsplittable or out of budget: whole reference to one side.
                let cen = (rf.min[axis] + rf.max[axis]) * 0.5;
                if cen < plane {
                    push(&mut l, &mut lg, &mut lc, rf.min, rf.max, rf.prim);
                } else {
                    push(&mut r, &mut rg, &mut rc, rf.min, rf.max, rf.prim);
                }
            }
        }
    }

    if l.is_empty() || r.is_empty() {
        return None;
    }
    Some(((l, lg, lc), (r, rg, rc)))
}

// ---------------------------------------------------------------------------
// Flatten
// ---------------------------------------------------------------------------

/// DFS flatten of the intermediate arena into the GPU node layout. Children
/// are emitted as adjacent pairs and whole subtrees stay contiguous, which is
/// both the shader contract and the cache-friendly order; leaf reference
/// ranges are re-emitted in DFS order, manufacturing the contiguous flat
/// triangle ranges (and resolving SBVH duplication). Also computes the
/// quality metrics and enforces the depth bound.
fn flatten(cfg: &BinSah, arena: &[BNode], refs: &[u32]) -> (Vec<BvhNode>, Vec<u32>, BuildStats) {
    let mut out: Vec<BvhNode> = Vec::with_capacity(arena.len());
    let mut order: Vec<u32> = Vec::with_capacity(refs.len());
    let mut stats = BuildStats::default();

    let root = &arena[0];
    let root_h = Bounds {
        min: root.min,
        max: root.max,
    }
    .half_area()
    .max(1e-30);

    let placeholder = BvhNode::interior(Vec3::ZERO, Vec3::ZERO, 0);
    out.push(placeholder);

    let mut sah = 0.0f32;
    let mut stack: Vec<(u32, u32, u32)> = Vec::with_capacity(2 * BVH_MAX_DEPTH as usize);
    stack.push((0, 0, 0));
    while let Some((src, dst, depth)) = stack.pop() {
        let node = &arena[src as usize];
        stats.max_depth = stats.max_depth.max(depth);
        let rel = Bounds {
            min: node.min,
            max: node.max,
        }
        .half_area()
            / root_h;
        if node.b != 0 {
            let first = order.len() as u32;
            let s = node.a as usize;
            order.extend_from_slice(&refs[s..s + node.b as usize]);
            out[dst as usize] = BvhNode::leaf(node.min.into(), node.max.into(), first, node.b);
            stats.leaf_count += 1;
            stats.max_leaf_size = stats.max_leaf_size.max(node.b);
            sah += rel * node.b as f32 * cfg.intersect_cost;
        } else {
            let left = out.len() as u32;
            out.push(placeholder);
            out.push(placeholder);
            out[dst as usize] = BvhNode::interior(node.min.into(), node.max.into(), left);
            sah += rel * cfg.traversal_cost;
            stack.push((node.a + 1, left + 1, depth + 1));
            stack.push((node.a, left, depth + 1));
        }
    }

    assert!(
        stats.max_depth <= BVH_MAX_DEPTH,
        "bvh depth {} exceeds shader stack bound {}",
        stats.max_depth,
        BVH_MAX_DEPTH
    );
    stats.node_count = out.len() as u32;
    stats.sah_cost = sah;
    stats.avg_leaf_size = order.len() as f32 / stats.leaf_count.max(1) as f32;
    (out, order, stats)
}

/// Permutes (and, with spatial splits, duplicates) the triangles into the
/// final leaf order — exactly one pass over the triangle data.
fn apply_ordering(triangles: &mut Vec<Triangle>, order: &[u32]) {
    let src = std::mem::take(triangles);
    *triangles = order.par_iter().map(|&i| src[i as usize]).collect();
}
