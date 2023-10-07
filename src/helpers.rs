pub fn setup_logging() {
    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {}] {}::{}\n- {}\n",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.target(),
                record.line().unwrap_or_default(),
                message
            ))
        })
        .level(log::LevelFilter::Debug)
        .level_for("wgpu", log::LevelFilter::Info)
        .level_for("wgpu_core", log::LevelFilter::Info)
        .chain(std::io::stdout())
        .apply()
        .unwrap();
}

pub fn create_file_recurse(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::File::create(path)
}

pub fn setup_tracing() -> Option<tracing_chrome::FlushGuard> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let local_time = chrono::Local::now();
    let formatted_time = local_time.format("%Y%m%d%H%M%S").to_string();

    let path_name = format!("./logs/learn-wgpu_{}.json", formatted_time);
    let path = std::path::Path::new(&path_name);
    create_file_recurse(path).unwrap();

    let _ = match std::fs::File::create(path) {
        Ok(file) => file,
        Err(e) => {
            // Handle any errors that occur when creating the file
            eprintln!("Error: {}", e);
            return None;
        }
    };

    let (chrome_layer, guard) = ChromeLayerBuilder::new().file(path).build();
    tracing_subscriber::registry().with(chrome_layer).init();

    Some(guard)
}
