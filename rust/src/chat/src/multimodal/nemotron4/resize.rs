// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! FP32 INTER_AREA resize, including OpenCV's area-emulation upsampling path.

pub(super) fn output_size(height: usize, width: usize, longest_edge: usize) -> (usize, usize) {
    let ratio = (height.max(width) as f64 / longest_edge as f64).max(1.0);
    let aligned = |dimension: usize| {
        ((dimension as f64 / ratio).round_ties_even() as usize).div_ceil(28) * 28
    };
    (aligned(height), aligned(width))
}

fn weights(source: usize, target: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = 1.0 / (target as f64 / source as f64);
    (0..target)
        .map(|destination| {
            let start = destination as f64 * scale;
            let end = (destination + 1) as f64 * scale;
            (start.floor() as usize..(end.ceil() as usize).min(source))
                .map(|index| {
                    let overlap = (end.min((index + 1) as f64) - start.max(index as f64)).max(0.0);
                    (index, (overlap / scale) as f32)
                })
                .collect()
        })
        .collect()
}

fn linear_area(source: usize, target: usize, destination: usize) -> (usize, usize, f32) {
    let inverse_scale = target as f64 / source as f64;
    let scale = 1.0 / inverse_scale;
    let first = ((destination as f64 * scale).floor() as usize).min(source - 1);
    // OpenCV rounds to FP32 before extracting the fractional part.
    let fraction = ((destination + 1) as f64 - (first + 1) as f64 * inverse_scale) as f32;
    let weight = if fraction <= 0.0 {
        0.0
    } else {
        fraction - fraction.floor()
    };
    (first, (first + 1).min(source - 1), weight)
}

pub(super) fn area(
    input: &[f32],
    source_height: usize,
    source_width: usize,
    height: usize,
    width: usize,
) -> Vec<f32> {
    if source_height == height && source_width == width {
        return input.to_vec();
    }
    let mut output = vec![0.0; height * width * 3];
    if source_height >= height && source_width >= width {
        let horizontal = weights(source_width, width);
        let vertical = weights(source_height, height);
        let mut row = vec![0.0_f32; width * 3];
        for (y, source_rows) in vertical.iter().enumerate() {
            for &(source_y, beta) in source_rows {
                row.fill(0.0);
                for (x, source_columns) in horizontal.iter().enumerate() {
                    for &(source_x, alpha) in source_columns {
                        for channel in 0..3 {
                            row[x * 3 + channel] +=
                                input[(source_y * source_width + source_x) * 3 + channel] * alpha;
                        }
                    }
                }
                for (x, value) in row.iter().enumerate() {
                    output[y * width * 3 + x] += value * beta;
                }
            }
        }
    } else {
        for y in 0..height {
            let (y0, y1, beta) = linear_area(source_height, height, y);
            for x in 0..width {
                let (x0, x1, alpha) = linear_area(source_width, width, x);
                for channel in 0..3 {
                    let at = |sy, sx| input[(sy * source_width + sx) * 3 + channel];
                    let upper = at(y0, x0) * (1.0 - alpha) + at(y0, x1) * alpha;
                    let lower = at(y1, x0) * (1.0 - alpha) + at(y1, x1) * alpha;
                    output[(y * width + x) * 3 + channel] = upper * (1.0 - beta) + lower * beta;
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_opencv_float32_area_reference() {
        let reference: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/nemotron4_resize.json"
        ))
        .unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let dim = |key: &str, index: usize| case[key][index].as_u64().unwrap() as usize;
            let (sh, sw, h, w) = (
                dim("source", 0),
                dim("source", 1),
                dim("target", 0),
                dim("target", 1),
            );
            let input = (0..sh * sw * 3)
                .map(|index| ((index * 37 + 19) % 256) as f32 / 255.0)
                .collect::<Vec<_>>();
            let actual = area(&input, sh, sw, h, w);
            let expected = case["expected"].as_array().unwrap();
            assert_eq!(actual.len(), expected.len());
            let max_error = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b.as_f64().unwrap() as f32).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_error < 4e-7,
                "{sh}x{sw} -> {h}x{w}: max error {max_error}"
            );
        }
    }

    #[test]
    fn follows_python_resize_boundaries() {
        let cases = [
            (29, 29, 56, 56),
            (56, 84, 56, 84),
            (1540, 1700, 952, 1036),
            (29, 1054, 56, 1036),
        ];
        for (h, w, expected_h, expected_w) in cases {
            assert_eq!(output_size(h, w, 1036), (expected_h, expected_w));
        }
    }
}
