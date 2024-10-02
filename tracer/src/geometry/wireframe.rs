use image::{RgbImage, RgbaImage};

use crate::{core::camera::Camera, utils::colour::Colour};



pub trait WireFrame {
    fn draw_wireframe(&self, img: &mut RgbaImage, colour: Colour, camera: &Camera);
}
