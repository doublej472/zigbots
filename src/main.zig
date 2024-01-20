const std = @import("std");
const core = @import("mach-core");
const gpu = core.gpu;

const Brain = @import("brain.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var allocator = gpa.allocator();

pub const App = @This();

// For every brain there should be BRAIN_WIDTH inputs and outputs
fn GPUTickBrain(brains: *const []Brain.Brain, inputs: *const []f32, outputs: *[]f32) !void {
    var raw_brain_data = try allocator.alloc(f32, Brain.GPU_BRAIN_STRIDE * brains.len);
    defer allocator.free(raw_brain_data);
    for (brains.*, 0..) |b, i| {
        const local_bdata = b.toGPUFormat();

        for (0..local_bdata.len) |j| {
            raw_brain_data[Brain.GPU_BRAIN_STRIDE * i + j] = local_bdata[j];
        }
    }

    const brain_data = core.device.createBuffer(&.{
        .usage = .{ .storage = true, .copy_dst = true },
        .size = @sizeOf(f32) * Brain.GPU_BRAIN_STRIDE * brains.len,
        .mapped_at_creation = .false,
    });
    defer brain_data.release();

    const brain_input = core.device.createBuffer(&.{
        .usage = .{ .storage = true, .copy_dst = true },
        .size = Brain.BRAIN_WIDTH * @sizeOf(f32) * brains.len,
        .mapped_at_creation = .false,
    });
    defer brain_input.release();

    const brain_output = core.device.createBuffer(&.{
        .usage = .{ .storage = true, .copy_src = true },
        .size = Brain.BRAIN_WIDTH * @sizeOf(f32) * brains.len,
        .mapped_at_creation = .false,
    });
    defer brain_output.release();

    const brain_output_staging = core.device.createBuffer(&.{
        .usage = .{ .map_read = true, .copy_dst = true },
        .size = Brain.BRAIN_WIDTH * @sizeOf(f32) * brains.len,
        .mapped_at_creation = .false,
    });
    defer brain_output_staging.release();

    const compute_module = core.device.createShaderModuleWGSL("compute.wgsl", @embedFile("compute.wgsl"));

    const compute_pipeline = core.device.createComputePipeline(&gpu.ComputePipeline.Descriptor{ .compute = gpu.ProgrammableStageDescriptor{
        .module = compute_module,
        .entry_point = "main",
    } });
    defer compute_pipeline.release();

    const layout = compute_pipeline.getBindGroupLayout(0);
    defer layout.release();

    const compute_bind_group = core.device.createBindGroup(&gpu.BindGroup.Descriptor.init(.{
        .layout = layout,
        .entries = &.{
            gpu.BindGroup.Entry.buffer(0, brain_data, 0, Brain.GPU_BRAIN_STRIDE * @sizeOf(f32) * brains.len),
            gpu.BindGroup.Entry.buffer(1, brain_input, 0, Brain.BRAIN_WIDTH * @sizeOf(f32) * brains.len),
            gpu.BindGroup.Entry.buffer(2, brain_output, 0, Brain.BRAIN_WIDTH * @sizeOf(f32) * brains.len),
        },
    }));
    defer compute_bind_group.release();

    compute_module.release();

    const encoder = core.device.createCommandEncoder(null);

    const wgcnt: u32 = @intCast(brains.len);

    const compute_pass = encoder.beginComputePass(null);
    compute_pass.setPipeline(compute_pipeline);
    compute_pass.setBindGroup(0, compute_bind_group, &.{});
    compute_pass.dispatchWorkgroups(wgcnt, 1, 1);
    compute_pass.end();
    compute_pass.release();

    encoder.copyBufferToBuffer(brain_output, 0, brain_output_staging, 0, Brain.BRAIN_WIDTH * @sizeOf(f32) * brains.len);

    var command = encoder.finish(null);
    encoder.release();

    var response: gpu.Buffer.MapAsyncStatus = undefined;
    const callback = (struct {
        pub inline fn callback(ctx: *gpu.Buffer.MapAsyncStatus, status: gpu.Buffer.MapAsyncStatus) void {
            ctx.* = status;
        }
    }).callback;

    var queue = core.queue;
    queue.writeBuffer(brain_data, 0, raw_brain_data);
    queue.writeBuffer(brain_input, 0, inputs.*);
    queue.submit(&[_]*gpu.CommandBuffer{command});
    command.release();

    brain_output_staging.mapAsync(.{ .read = true }, 0, Brain.BRAIN_WIDTH * @sizeOf(f32) * brains.len, &response, callback);
    while (true) {
        if (response == gpu.Buffer.MapAsyncStatus.success) {
            break;
        } else {
            core.device.tick();
        }
    }

    const staging_mapped = brain_output_staging.getConstMappedRange(f32, 0, Brain.BRAIN_WIDTH * brains.len);

    for (staging_mapped.?, 0..) |v, i| {
        outputs.*[i] = v;
    }
    brain_output_staging.unmap();
}

pub fn init(app: *App) !void {
    try core.init(.{
        .power_preference = .high_performance,
        .headless = true,
    });
    app.* = .{};

    var seed_val: u64 = undefined;
    try std.os.getrandom(std.mem.asBytes(&seed_val));
    var rand_source = std.rand.DefaultPrng.init(seed_val);
    var rng = rand_source.random();

    const nbrains = 2;
    var blist: [nbrains]Brain.Brain = undefined;
    for (&blist) |*b| {
        b.* = Brain.Brain.initRandom(&rng);
    }

    var binputs: [Brain.BRAIN_WIDTH * nbrains]f32 = .{1.0} ** (Brain.BRAIN_WIDTH * nbrains);
    var boutputs: [Brain.BRAIN_WIDTH * nbrains]f32 = .{1.0} ** (Brain.BRAIN_WIDTH * nbrains);

    for (blist, 0..) |b, i| {
        const bout = b.tick(binputs[i * Brain.BRAIN_WIDTH .. (i + 1) * Brain.BRAIN_WIDTH]);
        for (0..Brain.BRAIN_WIDTH) |j| {
            boutputs[i * Brain.BRAIN_WIDTH + j] = bout[j];
        }
    }
    std.debug.print("CPU outputs:\n", .{});
    for (boutputs) |o| {
        std.debug.print("{d} ", .{o});
    }
    std.debug.print("\n", .{});

    var slicestart: usize = 0;

    var binparr = binputs[slicestart..];
    var boutputs_gpu: [Brain.BRAIN_WIDTH * nbrains]f32 = .{1.0} ** (Brain.BRAIN_WIDTH * nbrains);
    var boutarr = boutputs_gpu[slicestart..];

    try GPUTickBrain(&blist[slicestart..], &binparr, &boutarr);

    std.debug.print("GPU output:\n", .{});
    for (boutarr) |o| {
        std.debug.print("{d} ", .{o});
    }
    std.debug.print("\n", .{});

    slicestart = 231;
}

pub fn deinit(app: *App) void {
    _ = app;
    defer _ = gpa.deinit();
    core.deinit();
}

pub fn update(_: *App) !bool {
    return true;
}
