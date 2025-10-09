use std::process::Command;

fn main() {
    println!("Running custom shader compilation script...");

    // For Windows, run the batch script
    #[cfg(target_os = "windows")]
    {
        let status = Command::new("cmd")
            .args(["/C", r"src\shaders\compile_shaders.bat"])
            .status()
            .expect("Could not run compile_shaders.bat on Windows");

        if !status.success() {
            panic!("Shader script execution failed on Windows!");
        }
    }

    // For macOS, run the shell script
    #[cfg(target_os = "macos")]
    {
        let status = Command::new("sh")
            .arg("./src/shaders/compile_shaders.sh")
            .status()
            .expect("Could not run compile_shaders.sh on macOS");

        if !status.success() {
            panic!("Shader script execution failed on macOS!");
        }
    }
}
