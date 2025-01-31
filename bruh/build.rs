use std::process::Command;

fn main() {
    println!("Running custom batch script...");

    let status = Command::new("cmd")
        .args(&["/C", "compile_shaders.bat"])
        .status()
        .expect("Could not build shaders atch script");

    if !status.success() {
        panic!("Could not build shaderscript execution failed!");
    }
}
