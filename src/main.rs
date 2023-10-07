fn main() {
    use learn_wgpu::helpers;

    // helpers::setup_logging();
    let _guard = helpers::setup_tracing();
    pollster::block_on(learn_wgpu::run());
}
