const BRAIN_LAYERS = 4u;
const BRAIN_WIDTH = 32u;

const GPU_BRAIN_BIAS_WIDTH = BRAIN_WIDTH;
const GPU_BRAIN_WEIGHT_WIDTH = BRAIN_WIDTH * BRAIN_WIDTH;
const GPU_BRAIN_STRIDE = (GPU_BRAIN_BIAS_WIDTH + GPU_BRAIN_WEIGHT_WIDTH) * BRAIN_LAYERS;

// GPUBrain layout, will be staged into GPU and never altered,
// only staged out, read, and deleted once done
// GPU_BRAIN_BIAS_WIDTH
// GPU_BRAIN_WEIGHT_WIDTH

@group(0) @binding(0) var<storage, read> brain_data: array<f32>;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<f32>;

@compute @workgroup_size(1) fn main(
    @builtin(global_invocation_id) id: vec3<u32>
) {
    var<function> offset = id.x;

    var<function> prev_internal_state: array<f32,BRAIN_WIDTH>;
    var<function> internal_state: array<f32,BRAIN_WIDTH>;

    // internal state starts with inputs
    for (var i: u32 = 0; i < BRAIN_WIDTH; i++) {
        prev_internal_state[i] = inputs[BRAIN_WIDTH * offset + i];
    }

    for (var layer_idx: u32 = 0; layer_idx < BRAIN_LAYERS; layer_idx++) {
        for (var bias_idx: u32 = 0; bias_idx < BRAIN_WIDTH; bias_idx++) {
            var sum: f32 = 0.0;

            for (var weight_idx: u32 = 0; weight_idx < BRAIN_WIDTH; weight_idx++) {
                sum += prev_internal_state[bias_idx] * brain_data[GPU_BRAIN_STRIDE * offset + (GPU_BRAIN_BIAS_WIDTH * 4) + (layer_idx * GPU_BRAIN_WEIGHT_WIDTH) + (bias_idx * GPU_BRAIN_BIAS_WIDTH) + weight_idx];
            }

            sum += brain_data[GPU_BRAIN_STRIDE * offset + layer_idx * GPU_BRAIN_BIAS_WIDTH + bias_idx];

            sum = clamp(sum, -1.0, 1.0);

            internal_state[bias_idx] = sum;
        }

        for (var i: u32 = 0; i < BRAIN_WIDTH; i++) {
            prev_internal_state[i] = internal_state[i];
        }
    }

    for (var i: u32 = 0; i < BRAIN_WIDTH; i++) {
        outputs[BRAIN_WIDTH * offset + i] = internal_state[i];
    }
}