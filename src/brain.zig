const std = @import("std");

pub const BRAIN_LAYERS: u64 = 4;
pub const BRAIN_WIDTH: u64 = 32;

pub const GPU_BRAIN_BIAS_WIDTH: u64 = BRAIN_WIDTH;
pub const GPU_BRAIN_WEIGHT_WIDTH: u64 = BRAIN_WIDTH * BRAIN_WIDTH;
pub const GPU_BRAIN_STRIDE: u64 = (GPU_BRAIN_BIAS_WIDTH + GPU_BRAIN_WEIGHT_WIDTH) * BRAIN_LAYERS;

pub const WEIGHT_RANGE: f32 = 0.7;
pub const BIAS_RANGE: f32 = 1.0;

pub const Brain = struct {
    weights: [GPU_BRAIN_WEIGHT_WIDTH * BRAIN_LAYERS]f32,
    biases: [GPU_BRAIN_BIAS_WIDTH * BRAIN_LAYERS]f32,

    pub fn initRandom(rand: *std.rand.Random) Brain {
        var b = Brain{
            .weights = .{0.5} ** 4096,
            .biases = .{1.0} ** 128,
        };

        for (&b.weights) |*weight| {
            weight.* = (rand.float(f32) * (WEIGHT_RANGE * 2)) - WEIGHT_RANGE;
        }

        for (&b.biases) |*bias| {
            bias.* = (rand.float(f32) * (BIAS_RANGE * 2)) - BIAS_RANGE;
        }

        return b;
    }

    pub fn toGPUFormat(b: *const Brain) [GPU_BRAIN_STRIDE]f32 {
        var data: [GPU_BRAIN_STRIDE]f32 = .{0.0} ** GPU_BRAIN_STRIDE;
        for (b.biases, 0..) |bias, i| {
            data[i] = bias;
        }

        for (b.weights, (GPU_BRAIN_BIAS_WIDTH * BRAIN_LAYERS)..) |weight, i| {
            data[i] = weight;
        }

        return data;
    }

    pub fn debugPrint(b: *const Brain) void {
        std.debug.print("Weights:\n", .{});
        for (b.weights) |w| {
            std.debug.print("{d} ", .{w});
        }

        std.debug.print("\n", .{});
        std.debug.print("Weights:\n", .{});
        for (b.biases) |w| {
            std.debug.print("{d} ", .{w});
        }
        std.debug.print("\n", .{});
    }

    pub fn tick(b: *const Brain, inputs: []f32) [BRAIN_WIDTH]f32 {
        var prev_internal_state: [BRAIN_WIDTH]f32 = .{0.0} ** BRAIN_WIDTH;
        var internal_state: [BRAIN_WIDTH]f32 = .{0.0} ** BRAIN_WIDTH;

        // internal state starts with inputs
        for (0..BRAIN_WIDTH) |i| {
            prev_internal_state[i] = inputs[i];
        }

        for (0..BRAIN_LAYERS) |layer_idx| {
            for (0..BRAIN_WIDTH) |bias_idx| {
                var sum: f32 = 0.0;

                for (0..BRAIN_WIDTH) |weight_idx| {
                    sum += prev_internal_state[bias_idx] * b.weights[layer_idx * GPU_BRAIN_WEIGHT_WIDTH + bias_idx * GPU_BRAIN_BIAS_WIDTH + weight_idx];
                }

                sum += b.biases[layer_idx * GPU_BRAIN_BIAS_WIDTH + bias_idx];

                sum = std.math.clamp(sum, -1.0, 1.0);

                internal_state[bias_idx] = sum;
            }

            for (0..BRAIN_WIDTH) |i| {
                prev_internal_state[i] = internal_state[i];
            }
        }

        return internal_state;
    }
};
