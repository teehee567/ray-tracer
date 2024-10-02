use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::Point3;
use ray_tracer::accelerators::aabb::AABB;
use ray_tracer::utils::random_f32_in;

fn generate_random_aabb() -> AABB {
    let min = Point3::new(
        random_f32_in(-100.0, 100.0),
        random_f32_in(-100.0, 100.0),
        random_f32_in(-100.0, 100.0),
    );
    let max = Point3::new(
        random_f32_in(min.x, min.x + 100.0),
        random_f32_in(min.y, min.y + 100.0),
        random_f32_in(min.z, min.z + 100.0),
    );

    AABB { min, max }
}

fn bench_combine_multiple(c: &mut Criterion) {
    let aabbs: Vec<AABB> = (0..10000).map(|_| generate_random_aabb()).collect();

    c.bench_function("AABB combine scalar", |b| {
        b.iter(|| {
            let mut combined = aabbs[0];
            for aabb in &aabbs[1..] {
                combined = AABB::combine_scalar(black_box(&combined), black_box(aabb));
            }
            combined
        })
    });

    c.bench_function("AABB combine sse", |b| {
        b.iter(|| {
            let mut combined = aabbs[0];
            for aabb in &aabbs[1..] {
                combined = AABB::combine_sse(black_box(&combined), black_box(aabb));
            }
            combined
        })
    });

    c.bench_function("AABB combine avx2", |b| {
        b.iter(|| {
            let mut combined = aabbs[0];
            for aabb in &aabbs[1..] {
                combined = AABB::combine_avx2(black_box(&combined), black_box(aabb));
            }
            combined
        })
    });


}

criterion_group!(benches, bench_combine_multiple,);
criterion_main!(benches);
