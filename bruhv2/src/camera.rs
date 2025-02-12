
use glam::Vec3;

pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
    pub first_pixel: Vec3,
    pub lens_radius: f32,
    pub fov: f32,
    pub focus_distance: f32,
    pub aspect_ratio: f32,
}

impl Camera {
    pub fn new(
        position: Vec3,
        target: Vec3,
        v_fov: f32,
        aspect_ratio: f32,
        aperture: f32,
        focus_dist: f32,
    ) -> Self {
        let forward = (target - position).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();

        let mut camera = Self {
            position,
            forward,
            right,
            up,
            horizontal: Vec3::ZERO,
            vertical: Vec3::ZERO,
            first_pixel: Vec3::ZERO,
            lens_radius: aperture / 2.0,
            fov: v_fov,
            focus_distance: focus_dist,
            aspect_ratio,
        };

        camera.set_aspect_ratio(aspect_ratio);
        camera
    }

    pub fn set_aspect_ratio(&mut self, ratio: f32) {
        self.aspect_ratio = ratio;
        let theta = self.fov.to_radians() / 2.0;
        let h = theta.tan();
        let viewport_height = 2.0 * h;
        let viewport_width = viewport_height * self.aspect_ratio;

        self.horizontal = self.focus_distance * viewport_width * self.right;
        self.vertical = self.focus_distance * viewport_height * self.up;

        self.first_pixel = self.position
            - self.horizontal / 2.0
            - self.vertical / 2.0
            + self.focus_distance * self.forward;
    }

    pub fn r#move(&mut self, v: Vec3) {
        self.position += v;
        self.first_pixel = self.position
            - self.horizontal / 2.0
            - self.vertical / 2.0
            + self.focus_distance * self.forward;
    }

    pub fn rotate_y(&mut self, theta: f32) {
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let original_x = self.forward.x;
        let original_z = self.forward.z;

        self.forward.x = original_x * cos_theta - original_z * sin_theta;
        self.forward.z = self.forward.x * sin_theta + original_z * cos_theta;
        self.forward = self.forward.normalize();

        self.right = self.forward.cross(Vec3::Y).normalize();
        self.up = self.right.cross(self.forward).normalize();

        let theta = self.fov.to_radians() / 2.0;
        let h = theta.tan();
        let viewport_height = 2.0 * h;
        let viewport_width = viewport_height * self.aspect_ratio;

        self.horizontal = self.focus_distance * viewport_width * self.right;
        self.vertical = self.focus_distance * viewport_height * self.up;

        self.first_pixel = self.position
            - self.horizontal / 2.0
            - self.vertical / 2.0
            + self.focus_distance * self.forward;
    }
}
