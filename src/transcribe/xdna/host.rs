//! The host side of the NPU encoder: everything between the device launches
//! (bf16 staging, im2col, layer norm, residuals). It follows whisper-xdna's
//! `encoder()` exactly; the tiny fixtures pin it down.

/// f32 -> bf16, round to nearest even (what numpy's `astype(bfloat16)` does).
#[inline]
pub fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Write `src` (rows x cols, row-major f32) into a zeroed-padded bf16 buffer
/// of `dst_cols` columns. Rows past `rows` and columns past `cols` are zeroed.
/// `ones_col` sets that column to 1.0 in the first `rows` rows (bias fold).
pub fn stage_bf16(
    dst: &mut [u16],
    dst_rows: usize,
    dst_cols: usize,
    src: &[f32],
    rows: usize,
    cols: usize,
    ones_col: Option<usize>,
) {
    debug_assert!(rows <= dst_rows && cols <= dst_cols);
    debug_assert!(dst.len() >= dst_rows * dst_cols && src.len() >= rows * cols);
    for r in 0..rows {
        let d = &mut dst[r * dst_cols..(r + 1) * dst_cols];
        let s = &src[r * cols..(r + 1) * cols];
        for (o, &v) in d[..cols].iter_mut().zip(s) {
            *o = bf16(v);
        }
        d[cols..].fill(0);
        if let Some(c) = ones_col {
            d[c] = bf16(1.0);
        }
    }
    dst[rows * dst_cols..dst_rows * dst_cols].fill(0);
}

/// `c[:rows, :cols]` of an (any, `c_cols`) f32 result, as a dense matrix.
pub fn unpad_f32(c: &[f32], c_cols: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        out.extend_from_slice(&c[r * c_cols..r * c_cols + cols]);
    }
    out
}

/// x (C, T) -> (T_out, C*kernel), columns ordered (channel, tap), matching
/// conv1d `weight.reshape(O, C*kernel)`.
pub fn im2col(x: &[f32], c: usize, t: usize, kernel: usize, stride: usize, pad: usize) -> Vec<f32> {
    let t_out = (t + 2 * pad - kernel) / stride + 1;
    let width = c * kernel;
    let mut out = vec![0f32; t_out * width];
    for to in 0..t_out {
        let row = &mut out[to * width..(to + 1) * width];
        for ch in 0..c {
            for j in 0..kernel {
                let ti = (to * stride + j) as isize - pad as isize;
                if ti >= 0 && (ti as usize) < t {
                    row[ch * kernel + j] = x[ch * t + ti as usize];
                }
            }
        }
    }
    out
}

/// Transpose a (rows, cols) row-major matrix.
pub fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = x[r * cols + c];
        }
    }
    out
}

/// Layer norm over the last axis, eps 1e-5, population variance.
pub fn layer_norm(x: &[f32], d: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (row, o) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = row
            .iter()
            .map(|&v| {
                let c = v as f64 - mean;
                c * c
            })
            .sum::<f64>()
            / d as f64;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..d {
            o[i] = ((row[i] as f64 - mean) * inv) as f32 * w[i] + b[i];
        }
    }
    out
}

/// `x[r, :] += v` for every row.
pub fn add_row(x: &mut [f32], v: &[f32]) {
    for row in x.chunks_exact_mut(v.len()) {
        for (a, &b) in row.iter_mut().zip(v) {
            *a += b;
        }
    }
}

pub fn add_into(x: &mut [f32], y: &[f32]) {
    for (a, &b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// Minimal reader for the little-endian f32 `.npy` fixtures.
#[cfg(test)]
pub fn read_npy_f32(path: &std::path::Path) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let (hlen, start) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen]).unwrap();
    assert!(header.contains("'descr': '<f4'"), "{}", header);
    assert!(header.contains("'fortran_order': False"), "{}", header);
    let shape_str = header.split("'shape': (").nth(1).unwrap();
    let shape_str = &shape_str[..shape_str.find(')').unwrap()];
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let data = bytes[start + hlen..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    (shape, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16(1.0), 0x3f80);
        assert_eq!(bf16(-2.0), 0xc000);
        // 1 + 2^-8 is exactly halfway between two bf16 values: ties to even
        assert_eq!(bf16(1.0 + 1.0 / 256.0), 0x3f80);
        // just above halfway rounds up
        assert_eq!(bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        // halfway with an odd lower bit rounds up to even
        assert_eq!(bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert!(bf16_to_f32(bf16(f32::NAN)).is_nan());
        assert_eq!(bf16_to_f32(bf16(3.140625)), 3.140625);
    }

    #[test]
    fn im2col_matches_reference_layout() {
        // C = 2 channels, T = 4 frames, kernel 3, stride 1, pad 1
        let x = [1., 2., 3., 4., 10., 20., 30., 40.];
        let cols = im2col(&x, 2, 4, 3, 1, 1);
        assert_eq!(cols.len(), 4 * 6);
        // frame 0: channel 0 taps (pad, 1, 2), channel 1 taps (pad, 10, 20)
        assert_eq!(&cols[0..6], &[0., 1., 2., 0., 10., 20.]);
        assert_eq!(&cols[18..24], &[3., 4., 0., 30., 40., 0.]);
        let s2 = im2col(&x, 2, 4, 3, 2, 1);
        assert_eq!(s2.len(), 2 * 6);
        assert_eq!(&s2[6..12], &[2., 3., 4., 20., 30., 40.]);
    }

    #[test]
    fn layer_norm_normalises_rows() {
        let x = [1., 2., 3., 4., 2., 2., 2., 2.];
        let out = layer_norm(&x, 4, &[1.; 4], &[0.; 4]);
        let mean: f32 = out[..4].iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-6);
        assert!(out[4..].iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn stage_pads_and_folds() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst = vec![0xffffu16; 3 * 4];
        stage_bf16(&mut dst, 3, 4, &src, 2, 2, Some(2));
        let f: Vec<f32> = dst.iter().map(|&h| bf16_to_f32(h)).collect();
        assert_eq!(f, vec![1., 2., 1., 0., 3., 4., 1., 0., 0., 0., 0., 0.]);
    }
}
