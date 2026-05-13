//! Host video source normalization for GStreamer playbin.

use std::path::{Path, PathBuf};

/// Returns `true` when `source` already has a URI scheme.
pub(super) fn is_uri_source(source: &str) -> bool {
    source.contains("://")
}

/// Converts a host source string into a playbin URI.
pub(super) fn source_uri(source: Option<&str>) -> Result<Option<String>, gstreamer::glib::Error> {
    let Some(source) = source else {
        return Ok(None);
    };
    if is_uri_source(source) {
        return Ok(Some(source.to_owned()));
    }
    gstreamer::glib::filename_to_uri(local_source_path(source), None)
        .map(|uri| Some(uri.to_string()))
}

/// Returns an absolute local path for GLib URI conversion when possible.
pub(super) fn local_source_path(source: &str) -> PathBuf {
    let path = Path::new(source);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(path),
        Err(_) => path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_source_yields_no_uri() {
        assert_eq!(source_uri(None).unwrap(), None);
    }

    #[test]
    fn uri_source_is_preserved_directly() {
        assert_eq!(
            source_uri(Some("https://example.invalid/movie.mp4")).unwrap(),
            Some(String::from("https://example.invalid/movie.mp4"))
        );
    }

    #[test]
    fn absolute_local_path_is_not_rebased() {
        let path = local_source_path("/tmp/renderide-video.mp4");

        assert_eq!(path, PathBuf::from("/tmp/renderide-video.mp4"));
    }
}
