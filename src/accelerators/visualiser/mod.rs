use glam::Vec3;
use smallvec::SmallVec;

use crate::accelerators::bvh_based::bvh::BvhNode;


pub struct AccelVis {
    pub nodes: Vec<AccelVisNode>,
    // pub stats: AccelVisStats,
}

impl AccelVis {
    // move to trait later on this is just ofr testing
    pub fn from_flat_bvh(nodes: &[BvhNode]) -> Self {
        let mut viz: Vec<AccelVisNode> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let (children, kind) = if node.is_leaf() {
                    (
                        SmallVec::new(),
                        AccelVisNodeKind::Leaf {
                            first_idx: node.idx(),
                            cnt: node.amt(),
                        },
                    )
                } else {
                    let left = node.idx();
                    (
                        SmallVec::from_slice(&[left, left + 1]),
                        AccelVisNodeKind::Inside,
                    )
                };

                AccelVisNode {
                    flat_index: i as u32,
                    parent: None,
                    children,
                    depth: 0,
                    bounds_min: node.min(),
                    bounds_max: node.max(),
                    split: None,
                    kind,
                }
            })
            .collect();

        // walk tree to construct depth
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            let (depth, children) = {
                let n = &viz[i];
                (n.depth, n.children.clone())
            };
            for child in children {
                let c = child as usize;
                viz[c].parent = Some(i as u32);
                viz[c].depth = depth + 1;
                stack.push(c);
            }
        }

        Self { nodes: viz }
    }
}

pub struct AccelVisNode {
    pub flat_index: u32,
    pub parent: Option<u32>,
    pub children: SmallVec<[u32; 8]>,
    pub depth: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub split: Option<SplitPlane>,
    pub kind: AccelVisNodeKind,
}

pub struct SplitPlane {
    pub axis: u8, //0=x, 1=y, 2=z
    pub position: f32,
}

pub enum AccelVisNodeKind {
    Inside,
    Leaf { first_idx: u32, cnt: u32},
}

pub enum AccelVisMode {
    LeavesOnly,
    DepthRange { low: u32, high: u32},
    TraversalHeatmap,
}


