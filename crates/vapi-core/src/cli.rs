//! Argument parsing for the two binaries.
//!
//! Hand-rolled rather than a dependency: there is one flag, and both binaries
//! have to agree on it. Deliberately strict — an argument it does not
//! recognise is an error naming the argument, not something silently ignored,
//! for the same reason the config structs are `deny_unknown_fields`. A
//! misspelled flag that appears to work is worse than one that fails.

use std::ffi::OsString;
use std::path::PathBuf;

use crate::{Error, Result};

/// What the command line asked for.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Args {
    /// `--config <path>`. Takes precedence over `$VAPI_CONFIG`, which in turn
    /// takes precedence over `./vapi.toml`.
    pub config: Option<PathBuf>,
    /// `--help`. The caller prints its own usage and exits, rather than this
    /// module exiting the process, so that the parser stays testable.
    pub help: bool,
}

/// Parse arguments, which should not include the program name.
pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Args> {
    let mut out = Args::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let text = arg.to_string_lossy().into_owned();
        match text.as_str() {
            "--help" | "-h" => out.help = true,
            "--config" | "-c" => {
                let value = args.next().ok_or_else(|| {
                    Error::Config(format!("{text} needs a path, e.g. --config vapi.toml"))
                })?;
                out.config = Some(PathBuf::from(value));
            }
            _ if text.starts_with("--config=") => {
                let value = text.trim_start_matches("--config=");
                if value.is_empty() {
                    return Err(Error::Config(
                        "--config= needs a path, e.g. --config=vapi.toml".into(),
                    ));
                }
                out.config = Some(PathBuf::from(value));
            }
            _ => {
                return Err(Error::Config(format!(
                    "unknown argument {text:?}; the only flags are --config <path> and --help"
                )));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> Result<Args> {
        parse(args.iter().map(OsString::from))
    }

    #[test]
    fn no_arguments_asks_for_nothing() {
        assert_eq!(run(&[]).unwrap(), Args::default());
    }

    #[test]
    fn the_path_can_be_separate_or_joined() {
        for form in [
            vec!["--config", "configs/laya.toml"],
            vec!["--config=configs/laya.toml"],
            vec!["-c", "configs/laya.toml"],
        ] {
            let got = run(&form).unwrap();
            assert_eq!(
                got.config,
                Some(PathBuf::from("configs/laya.toml")),
                "{form:?}"
            );
        }
    }

    #[test]
    fn a_flag_with_no_path_is_an_error_that_says_so() {
        // Otherwise it silently falls back to vapi.toml and serves the wrong
        // model, which looks like a bug somewhere else entirely.
        for form in [vec!["--config"], vec!["-c"], vec!["--config="]] {
            let err = run(&form).unwrap_err().to_string();
            assert!(err.contains("needs a path"), "{form:?}: {err}");
        }
    }

    #[test]
    fn an_unknown_argument_names_itself() {
        let err = run(&["--confgi", "x.toml"]).unwrap_err().to_string();
        assert!(err.contains("--confgi"), "{err}");
        assert!(err.contains("--config"), "{err}");
    }

    #[test]
    fn help_is_recognised_in_both_spellings() {
        assert!(run(&["--help"]).unwrap().help);
        assert!(run(&["-h"]).unwrap().help);
    }

    #[test]
    fn a_path_that_is_not_utf8_still_works() {
        // Paths are `OsString` all the way through; only the flag itself is
        // read as text.
        let args = vec![
            OsString::from("--config"),
            OsString::from("cfg/ünïcode.toml"),
        ];
        assert_eq!(
            parse(args).unwrap().config,
            Some(PathBuf::from("cfg/ünïcode.toml"))
        );
    }
}
