use ezpc::*;
use std::str::FromStr;

use crate::error::ShapeDecompressionError;

pub fn decompress_shape(
    samples: Vec<f64>,
    num_samples: u32,
) -> Result<Vec<f64>, ShapeDecompressionError> {
    // First, decompress into the deriviate of the shape
    let mut deriv = Vec::with_capacity(num_samples as usize);

    // The two samples before the current one, to detect RLE
    let mut a = f64::NAN;
    let mut b = f64::NAN;
    // After a detected RLE, skip the RLE check for two samples
    let mut skip = 0;

    for (index, sample) in samples.into_iter().enumerate() {
        if a == b && skip == 0 {
            if sample != sample.round() {
                Err(ShapeDecompressionError::RleCountIsNotInteger {
                    index,
                    value: sample,
                })?;
            }

            skip = 2;
            for _ in 0..sample as usize {
                deriv.push(b);
            }
        } else {
            if skip > 0 {
                skip -= 1;
            }
            deriv.push(sample);
        }

        a = b;
        b = sample;
    }

    if deriv.len() != num_samples as usize {
        Err(ShapeDecompressionError::WrongDecompressedCount {
            count: deriv.len(),
            expected: num_samples as usize,
        })?;
    }

    // Then, do a cumultative sum to get the shape
    Ok(deriv
        .into_iter()
        .scan(0.0, |acc, x| {
            *acc += x;
            Some(*acc)
        })
        .collect())
}

// Simple parsers that are not really specific to pulseq

/// Matches at least one whitespace but now newline
pub fn ws() -> Matcher<impl Match> {
    one_of(" \t").repeat(1..)
}

/// Matches as many whitespaces and comments as possible but expects at least one '\n'
pub fn nl() -> Matcher<impl Match> {
    let ignore = || ws() | (tag("#") + none_of("\n").repeat(0..));
    let eol = || tag("\r\n") | tag("\n");

    eof() | ((ignore().opt() + eol()).repeat(1..) + ignore().opt())
}

/// Shorthand for tag + whitespace
pub fn tag_ws(tag_str: &'static str) -> Matcher<impl Match> {
    tag(tag_str) + ws()
}

/// Shorthand for tag + newline
pub fn tag_nl(tag_str: &'static str) -> Matcher<impl Match> {
    tag(tag_str) + nl()
}

pub fn ident() -> Parser<impl Parse<Output = String>> {
    is_a(char::is_alphanumeric)
        .repeat(1..)
        .map(|s| s.to_owned())
}

pub fn int() -> Parser<impl Parse<Output = u32>> {
    (tag("0") | (one_of("123456789") + one_of("0123456789").repeat(0..)))
        .convert(|s| s.parse(), "Failed to parse string as int")
}

pub fn float() -> Parser<impl Parse<Output = f64>> {
    let integer = tag("0") | (one_of("123456789") + one_of("0123456789").repeat(0..));
    let frac = tag(".") + one_of("0123456789").repeat(1..);
    let exp = one_of("eE") + one_of("+-").opt() + one_of("0123456789").repeat(1..);
    let number = tag("-").opt() + integer + frac.opt() + exp.opt();
    number.convert(f64::from_str, "Failed to parse string as float")
}
