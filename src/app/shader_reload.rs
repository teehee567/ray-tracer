use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, bounded};
use notify_debouncer_mini::notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_mini::{DebounceEventResult, Debouncer, new_debouncer};

use crate::app::render_controller::RenderCommand;
use crate::gui::PushGui;

/// Freshly compiled path tracer SPIR-V on its way to the render thread.
#[derive(Clone)]
pub struct ShaderBlob(pub Arc<Vec<u8>>);

impl fmt::Debug for ShaderBlob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ShaderBlob({} bytes)", self.0.len())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ReloadRequest {
    Manual,
    FileChanged,
}

/// Hot reload for the path tracer compute shader: watches `src/shaders`,
/// recompiles `main.comp` with glslc off-thread, hands successful builds to
/// the render thread and compile errors to the GUI. The embedded SPIR-V from
/// build time remains the startup pipeline; this only swaps it afterwards.
pub struct ShaderReloader {
    request_tx: Sender<ReloadRequest>,
    // Kept alive for the lifetime of the app; dropping stops the watcher.
    _debouncer: Option<Debouncer<RecommendedWatcher>>,
}

impl ShaderReloader {
    pub fn spawn(command_tx: Sender<RenderCommand>, gui_tx: Sender<PushGui>) -> Self {
        let (request_tx, request_rx) = bounded::<ReloadRequest>(16);

        let shader_dir = find_shader_dir();
        let debouncer = shader_dir
            .as_deref()
            .and_then(|dir| start_watcher(dir, request_tx.clone()));

        // The worker exits when every request sender is dropped; no join
        // needed for an app-lifetime helper.
        let _ = thread::Builder::new()
            .name("shader-reload".into())
            .spawn(move || reload_worker(request_rx, shader_dir, command_tx, gui_tx));

        Self {
            request_tx,
            _debouncer: debouncer,
        }
    }

    pub fn request_sender(&self) -> Sender<ReloadRequest> {
        self.request_tx.clone()
    }
}

fn find_shader_dir() -> Option<PathBuf> {
    let candidates = [
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders"),
        PathBuf::from("./src/shaders"),
    ];
    let found = candidates
        .into_iter()
        .find(|dir| dir.join("main.comp").is_file());
    if found.is_none() {
        log::warn!("src/shaders not found; shader hot reload disabled");
    }
    found
}

fn start_watcher(
    dir: &Path,
    request_tx: Sender<ReloadRequest>,
) -> Option<Debouncer<RecommendedWatcher>> {
    let debouncer = new_debouncer(
        Duration::from_millis(300),
        move |result: DebounceEventResult| {
            let Ok(events) = result else { return };
            // GLSL sources only; ignores the .spv files build.rs drops
            // into the same directory.
            let relevant = events.iter().any(|event| {
                matches!(
                    event.path.extension().and_then(|ext| ext.to_str()),
                    Some("comp" | "glsl")
                )
            });
            if relevant {
                let _ = request_tx.try_send(ReloadRequest::FileChanged);
            }
        },
    );

    match debouncer {
        Ok(mut debouncer) => match debouncer.watcher().watch(dir, RecursiveMode::Recursive) {
            Ok(()) => Some(debouncer),
            Err(err) => {
                log::warn!(
                    "failed to watch {}: {err}; hot reload via button only",
                    dir.display()
                );
                None
            }
        },
        Err(err) => {
            log::warn!("failed to create shader watcher: {err}; hot reload via button only");
            None
        }
    }
}

fn reload_worker(
    request_rx: Receiver<ReloadRequest>,
    shader_dir: Option<PathBuf>,
    command_tx: Sender<RenderCommand>,
    gui_tx: Sender<PushGui>,
) {
    while let Ok(request) = request_rx.recv() {
        // coalesce a burst of requests into a single compile
        while request_rx.try_recv().is_ok() {}

        let Some(dir) = shader_dir.as_deref() else {
            let _ = gui_tx.try_send(PushGui::ShaderReload {
                error: Some("shader source directory not found".into()),
            });
            continue;
        };

        log::info!("recompiling main.comp ({request:?})");
        match compile_main_comp(dir) {
            Ok(spv) => {
                let _ = command_tx.send(RenderCommand::ReloadShader(ShaderBlob(Arc::new(spv))));
            }
            Err(err) => {
                log::warn!("shader compile failed:\n{err}");
                let _ = gui_tx.try_send(PushGui::ShaderReload { error: Some(err) });
            }
        }
    }
}

fn compile_main_comp(shader_dir: &Path) -> Result<Vec<u8>, String> {
    let glslc = std::env::var_os("VULKAN_SDK")
        .map(|sdk| PathBuf::from(sdk).join("bin/glslc"))
        .filter(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from("glslc"));

    let out_path =
        std::env::temp_dir().join(format!("raytracer-main-{}.comp.spv", std::process::id()));

    let output = Command::new(&glslc)
        .arg(shader_dir.join("main.comp"))
        .arg("-o")
        .arg(&out_path)
        .output()
        .map_err(|err| format!("failed to run {}: {err}", glslc.display()))?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }

    let spv = std::fs::read(&out_path)
        .map_err(|err| format!("failed to read {}: {err}", out_path.display()))?;
    let _ = std::fs::remove_file(&out_path);

    // guard before the bytes ever reach vkCreateShaderModule
    let valid = spv.len() >= 4 && spv.len() % 4 == 0 && spv[..4] == 0x0723_0203u32.to_le_bytes();
    if !valid {
        return Err("glslc produced invalid SPIR-V output".into());
    }
    Ok(spv)
}
