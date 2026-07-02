use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, bounded};
use notify_debouncer_mini::notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_mini::{DebounceEventResult, Debouncer, new_debouncer};

use crate::app::render_controller::RenderCommand;
use crate::app::shader_reload::ReloadRequest;
use crate::gui::PushGui;
use crate::scene::Scene;

pub struct SceneReloader {
    request_tx: Sender<ReloadRequest>,
    _debouncer: Option<Debouncer<RecommendedWatcher>>,
}

impl SceneReloader {
    pub fn spawn(
        scene_path: PathBuf,
        command_tx: Sender<RenderCommand>,
        gui_tx: Sender<PushGui>,
    ) -> Self {
        let (request_tx, request_rx) = bounded::<ReloadRequest>(16);

        let debouncer = scene_path
            .parent()
            .and_then(|dir| start_watcher(dir, request_tx.clone()));

        let _ = thread::Builder::new()
            .name("scene-reload".into())
            .spawn(move || reload_worker(request_rx, scene_path, command_tx, gui_tx));

        Self {
            request_tx,
            _debouncer: debouncer,
        }
    }

    pub fn request_sender(&self) -> Sender<ReloadRequest> {
        self.request_tx.clone()
    }
}

fn start_watcher(
    dir: &Path,
    request_tx: Sender<ReloadRequest>,
) -> Option<Debouncer<RecommendedWatcher>> {
    let debouncer = new_debouncer(
        Duration::from_millis(300),
        move |result: DebounceEventResult| {
            let Ok(events) = result else { return };
            let relevant = events.iter().any(|event| {
                matches!(
                    event.path.extension().and_then(|ext| ext.to_str()),
                    Some("yaml" | "yml" | "obj")
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
                    "failed to watch {}: {err}; scene reload via button only",
                    dir.display()
                );
                None
            }
        },
        Err(err) => {
            log::warn!("failed to create scene watcher: {err}; scene reload via button only");
            None
        }
    }
}

fn reload_worker(
    request_rx: Receiver<ReloadRequest>,
    scene_path: PathBuf,
    command_tx: Sender<RenderCommand>,
    gui_tx: Sender<PushGui>,
) {
    while let Ok(request) = request_rx.recv() {
        while request_rx.try_recv().is_ok() {}

        log::info!("reloading scene {} ({request:?})", scene_path.display());
        match load_scene(&scene_path) {
            Ok(scene) => {
                let _ = command_tx.send(RenderCommand::ReloadScene(Box::new(scene)));
            }
            Err(err) => {
                log::warn!("scene reload failed:\n{err}");
                let _ = gui_tx.try_send(PushGui::SceneReload { error: Some(err) });
            }
        }
    }
}

fn load_scene(path: &Path) -> Result<Scene, String> {
    let path_str = path
        .to_str()
        .ok_or_else(|| format!("scene path is not valid UTF-8: {}", path.display()))?;

    let result = catch_unwind(AssertUnwindSafe(|| Scene::from_new(path_str)));
    match result {
        Ok(Ok(scene)) => Ok(scene),
        Ok(Err(err)) => Err(err.to_string()),
        Err(panic) => Err(panic_message(panic)),
    }
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = panic.downcast_ref::<String>() {
        msg.clone()
    } else if let Some(msg) = panic.downcast_ref::<&str>() {
        (*msg).to_string()
    } else {
        "scene loader panicked".to_string()
    }
}
