//! A fixture with one function and one test, so a task can ask a model to
//! read the file and say what it does.

/// The greeting this crate exists to produce.
pub fn greeting() -> &'static str {
    "hello from the fixture"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_greeting_is_the_expected_one() {
        assert_eq!(greeting(), "hello from the fixture");
    }
}
