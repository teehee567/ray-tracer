use std::{ffi::c_void, mem};


const INITIAL_SIZE: usize = 16;

#[derive(Debug)]
pub struct BufferBuilder {
    buffer: Vec<u8>,
}

impl BufferBuilder {
    pub fn new() -> Self {
        BufferBuilder {
            buffer: Vec::with_capacity(INITIAL_SIZE),
        }
    }

    pub fn align_to_16(&mut self) {
        let remainder = self.buffer.len() % 16;
        if remainder != 0 {
            let pad_size = 16 - remainder;
            self.pad(pad_size);
        }
    }

    pub fn get_offset(&self) -> usize {
        self.buffer.len()
    }

    pub fn pad(&mut self, amt: usize) {
        self.buffer.resize(self.buffer.len() + amt, 0);
    }

    pub fn append<T: Copy>(&mut self, value: T) {
        let size = mem::size_of::<T>();
        // SAFETY: The caller must guarantee that T is a POD type.
        let value_bytes =
            unsafe { std::slice::from_raw_parts((&value as *const T) as *const u8, size) };
        self.buffer.extend_from_slice(value_bytes);
    }

    pub unsafe fn write(&self, output: *mut c_void) {
        std::ptr::copy_nonoverlapping(self.buffer.as_ptr(), output as *mut u8, self.buffer.len());
    }
}
