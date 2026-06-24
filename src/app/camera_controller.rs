use glam::{Mat3, Mat4, Vec2, Vec3};

use crate::types::CameraBufferObject;

#[derive(Clone, Copy, Debug, Default)]
pub struct CameraInput {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub look_delta: Vec2,
    pub scroll: f32,
}

pub struct CameraController {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    move_speed: f32,
    look_sensitivity: f32,
}

const MIN_SPEED: f32 = 0.01;
const MAX_SPEED: f32 = 1000.0;
const PITCH_LIMIT: f32 = 89.0 * std::f32::consts::PI / 180.0;

impl CameraController {
    pub fn from_camera(camera: &CameraBufferObject) -> Self {
        let position = camera.location.0;
        let mut forward = (Mat3::from_mat4(camera.rotation.0) * Vec3::NEG_Z).normalize_or_zero();
        if forward == Vec3::ZERO {
            forward = Vec3::NEG_Z;
        }
        Self {
            position,
            yaw: forward.x.atan2(-forward.z),
            pitch: forward.y.clamp(-1.0, 1.0).asin(),
            move_speed: 0.3,
            look_sensitivity: 0.0025,
        }
    }

    fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, sp, -cp * cy)
    }

    fn rotation(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y).transpose()
    }

    pub fn update(&mut self, input: CameraInput, dt: f32) -> Option<(Vec3, Mat4)> {
        if input.scroll != 0.0 {
            self.move_speed =
                (self.move_speed * 1.1_f32.powf(input.scroll)).clamp(MIN_SPEED, MAX_SPEED);
        }

        let mut changed = false;

        if input.look_delta != Vec2::ZERO {
            self.yaw += input.look_delta.x * self.look_sensitivity;
            self.pitch = (self.pitch - input.look_delta.y * self.look_sensitivity)
                .clamp(-PITCH_LIMIT, PITCH_LIMIT);
            changed = true;
        }

        let forward = self.forward();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let mut motion = Vec3::ZERO;
        if input.forward {
            motion += forward;
        }
        if input.back {
            motion -= forward;
        }
        if input.right {
            motion += right;
        }
        if input.left {
            motion -= right;
        }
        if input.up {
            motion += Vec3::Y;
        }
        if input.down {
            motion -= Vec3::Y;
        }

        if motion != Vec3::ZERO {
            self.position += motion.normalize() * self.move_speed * dt.max(0.0);
            changed = true;
        }

        changed.then(|| (self.position, self.rotation()))
    }
}
