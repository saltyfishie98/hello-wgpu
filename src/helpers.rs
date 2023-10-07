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
