use std::{ffi::c_void, mem};

use anyhow::{Result, anyhow};

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

    pub fn append_with_size<T: Copy>(&mut self, value: T, size: usize) {
        let type_size = mem::size_of::<T>();
        let value_bytes =
            unsafe { std::slice::from_raw_parts((&value as *const T) as *const u8, type_size) };
        self.buffer.extend_from_slice(value_bytes);
        if size > type_size {
            self.buffer
                .resize(self.buffer.len() + (size - type_size), 0);
        } else if size < type_size {
            let new_len = self.buffer.len() - (type_size - size);
            self.buffer.truncate(new_len);
        }
    }

    pub fn get_relative_offset<T>(&self) -> Result<usize> {
        let type_size = mem::size_of::<T>();
        if self.buffer.len() == 0 {
            return Ok(0);
        }
        if self.buffer.len() % type_size != 0 {
            return Err(anyhow!(
                "Trying to get offset with regards to type not granular enough"
            ));
        }
        Ok(self.buffer.len() / type_size)
    }

    pub unsafe fn write(&self, output: *mut c_void) {
        std::ptr::copy_nonoverlapping(self.buffer.as_ptr(), output as *mut u8, self.buffer.len());
    }
}
