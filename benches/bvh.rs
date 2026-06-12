use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use raytracer::accelerators::bvh::Bvh;
use raytracer::accelerators::Accelerator;
use raytracer::scene::Scene;
use raytracer::Material;
use std::hint::black_box;

const SPONZA: &str = "scenes/sponza/Sponza.gltf";

fn bench_bvh_build(c: &mut Criterion) {
    let scene = Scene::from_gltf(SPONZA).expect("failed to load sponza");
    let triangles = scene.components.triangles;
    println!("loaded {} triangles from {SPONZA}", triangles.len());

    let mut group = c.benchmark_group("bvh_build");
    group.sample_size(10);
    group.throughput(Throughput::Elements(triangles.len() as u64));

    group.bench_function("sponza", |b| {
        b.iter(|| {
            let mut triangles = triangles.clone();
            let mut materials = vec![Material::default()];
            let nodes =
                Bvh::default().build(black_box(&mut triangles), black_box(&mut materials));
            black_box(nodes.len())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_bvh_build);
criterion_main!(benches);
