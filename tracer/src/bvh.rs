use std::{cmp::Ordering, sync::Arc};

use crate::{
    aabb::AABB,
    hittable::{HitRecord, Hittable},
    interval::Interval,
    ray::Ray,
};

enum BVHNode {
    Branch { left: Box<BVH>, right: Box<BVH> },
    Leaf(Box<dyn Hittable>),
}

pub struct BVH {
    tree: BVHNode,
    bbox: AABB,
}

impl BVH {
    pub fn new(mut objects: Vec<Box<dyn Hittable>>, time0: f32, time1: f32) -> Self {
        fn axis_range(
            hittable: &[Box<dyn Hittable>],
            time0: f32,
            time1: f32,
            axis: usize,
        ) -> f32 {
            let (min, max) = hittable
                .iter()
                .fold((f32::MAX, f32::MIN), |(bmin, bmax), hit| {
                    let aabb = hit.bounding_box();
                    let (axis_min, axis_max) = match axis {
                        0 => (aabb.min.x, aabb.max.x),
                        1 => (aabb.min.y, aabb.max.y),
                        2 => (aabb.min.z, aabb.max.z),
                        _ => unreachable!(),
                    };
                    (bmin.min(axis_min), bmax.max(axis_max))
                });
            max - min
        }

        let longest_axis = (0..3)
            .map(|a| (a, axis_range(&objects, time0, time1, a)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(axis, _)| axis)
            .unwrap_or(0);

        // Sort hittables along the chosen axis
        objects.sort_unstable_by(|a, b| {
            let abb = a.bounding_box();
            let bbb = b.bounding_box();
            let ac = match longest_axis {
                0 => abb.min.x + abb.max.x,
                1 => abb.min.y + abb.max.y,
                2 => abb.min.z + abb.max.z,
                _ => panic!("Axis out of range(it might be over)"),
            };
            let bc = match longest_axis {
                0 => bbb.min.x + bbb.max.x,
                1 => bbb.min.y + bbb.max.y,
                2 => bbb.min.z + bbb.max.z,
                _ => panic!("Axis out of range(it might be over)"),
            };
            ac.partial_cmp(&bc).unwrap()
        });

        let len = objects.len();
        match len {
            0 => panic!("No elements in scene"),
            1 => {
                let leaf = objects.pop().unwrap();
                let bbox = leaf.bounding_box().clone();
                BVH {
                    tree: BVHNode::Leaf(leaf),
                    bbox,
                }
            }
            _ => {
                let right = BVH::new(objects.drain(len / 2..).collect(), time0, time1);
                let left = BVH::new(objects, time0, time1);
                let bbox = AABB::combine(&left.bbox, &right.bbox);
                BVH {
                    tree: BVHNode::Branch {
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    bbox,
                }
            }
        }
    }
}

impl Hittable for BVH {
    fn hit(&self, ray: &Ray, mut ray_t: Interval, rec: &mut HitRecord) -> bool {
        if self.bbox.hit(ray, ray_t) {
            match &self.tree {
                BVHNode::Leaf(leaf) => leaf.hit(ray, ray_t, rec),
                BVHNode::Branch { left, right } => {
                    let hit_left = left.hit(ray, ray_t, rec);
                    let hit_right = right.hit(ray, Interval::new(ray_t.min, if hit_left { rec.t } else { ray_t.max }), rec);

                    hit_left || hit_right
                }
            }
        } else {
            false
        }
    }

    fn bounding_box(&self) -> &AABB {
        &self.bbox
    }
}