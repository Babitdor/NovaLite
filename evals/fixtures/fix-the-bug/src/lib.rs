//! A fixture with a bug and a test that catches it.
//!
//! The bug is a subtraction where an addition belongs, which is the kind of
//! thing a model can only find by reading the code and running the test: the
//! test names the symptom, not the cause.

/// Add two numbers.
pub fn add(a: i32, b: i32) -> i32 {
    a - b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_and_three_make_five() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    fn zero_is_the_identity() {
        assert_eq!(add(7, 0), 7);
    }
}
