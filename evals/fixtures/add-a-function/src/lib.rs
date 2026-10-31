//! A fixture with one function, so a task can ask a model to add another and
//! prove it works.

/// Add two numbers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_and_three_make_five() {
        assert_eq!(add(2, 3), 5);
    }
}
