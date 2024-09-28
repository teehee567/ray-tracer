use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{Point3, Vector3};
use ray_tracer::core::interval::Interval;
use ray_tracer::core::ray::Ray;
use ray_tracer::geometry::objects::mesh::intersect_triangle;

pub fn intersect_triangle_benchmark(c: &mut Criterion) {
    // Define the triangle vertices
    let tri_a = Vector3::new(0.0, 0.0, 0.0);
    let tri_b = Vector3::new(1.0, 0.0, 0.0);
    let tri_c = Vector3::new(0.0, 1.0, 0.0);

    // Generate a list of rays with varying origins and directions
    let rays: Vec<Ray> = (0..1000)
        .map(|i| {
            let x = (i as f32) / 1000.0;
            let y = (i as f32) / 1000.0;
            let ray_origin = Point3::new(x, y, -1.0);
            let ray_direction = Vector3::new(0.0, 0.0, 1.0);
            Ray::new(ray_origin, ray_direction)
        })
        .collect();

    let ray_t = Interval {
        min: 0.0,
        max: f32::INFINITY,
    };

    c.bench_function("intersect_triangle_multiple_rays", |b| {
        b.iter(|| {
            for ray in &rays {
                let _ = intersect_triangle(
                    black_box(ray),
                    black_box(ray_t),
                    black_box(tri_a),
                    black_box(tri_b),
                    black_box(tri_c),
                );
            }
        });
    });
}

criterion_group!(benches, intersect_triangle_benchmark,);
criterion_main!(benches);
