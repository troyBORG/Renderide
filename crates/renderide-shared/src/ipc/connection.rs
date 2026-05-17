//! Command-line connection parameters for Cloudtoid IPC (`-QueueName` / `-QueueCapacity`).
//!
//! Matches the managed host's argument convention (see `RenderingManager.GetConnectionParameters`).

use std::env;
use std::io::{Cursor, Read};
use std::sync::atomic::{AtomicBool, Ordering};

use thiserror::Error;

/// Error returned when renderer initialization fails (singleton or IPC connect).
#[derive(Debug, Error)]
pub enum InitError {
    /// Only one renderer session may initialize the singleton guard.
    #[error("renderer singleton already initialized")]
    SingletonAlreadyExists,
    /// Opening a subscriber or publisher failed.
    #[error("IPC connect: {0}")]
    IpcConnect(String),
}

/// Default queue capacity (8 MiB), matching `MessagingManager.DEFAULT_CAPACITY`.
pub const DEFAULT_QUEUE_CAPACITY: i64 = 8_388_608;

/// Parsed connection parameters for IPC with the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectionParams {
    /// Base queue name (without `Primary`/`Background` or `A`/`S` suffixes).
    pub queue_name: String,
    /// Ring capacity in bytes (user payload; excludes queue header).
    pub queue_capacity: i64,
}

/// Process-wide guard ensuring only one renderer initializes the IPC singleton.
static RENDERIDE_SINGLETON_CLAIMED: AtomicBool = AtomicBool::new(false);

/// Reserves the single-renderer process guard (Unity: one `RenderingManager`).
///
/// Call once at startup; subsequent calls return [`InitError::SingletonAlreadyExists`].
pub fn try_claim_renderer_singleton() -> Result<(), InitError> {
    if RENDERIDE_SINGLETON_CLAIMED.swap(true, Ordering::SeqCst) {
        return Err(InitError::SingletonAlreadyExists);
    }
    Ok(())
}

/// Parses `-QueueName` / `-QueueCapacity` from `std::env::args`, if present.
///
/// If `-AttachRenderer` was passed, instead blocks and listens on UDP
/// port 42512 for a message from the engine to provide these.
///
/// Returns [`None`] when arguments are missing or invalid so the process can run without IPC.
pub fn get_connection_parameters() -> Option<ConnectionParams> {
    let args: Vec<String> = env::args().collect();

    if args.iter().any(|arg| arg.ends_with("-AttachRenderer")) {
        // Wait for a UDP packet with the connection parameters instead of parsing from command-line args.
        logger::info!("Attempting to wait for Resonite to attach debug renderer.");
        return get_connection_parameters_from_udp();
    }
    parse_connection_args(&args)
}

/// Wait for connection parameters by listening on UDP port 42512.
fn get_connection_parameters_from_udp() -> Option<ConnectionParams> {
    // The data is encoded in the format of dotnet's BinaryWriter:
    // QueueName: length-prefixed UTF-8 (7-bit encoded i32 length, then bytes).
    // QueueCapacity: 8-byte little-endian i64.

    // Get data
    let socket = std::net::UdpSocket::bind("127.0.0.1:42512").ok()?;
    logger::info!("Listening on UDP port 42512. Launch Resonite with -AttachRenderer");
    let mut buf = [0u8; 1024];
    let (len, _) = socket.recv_from(&mut buf).ok()?;
    let mut cursor = Cursor::new(&buf[..len]);

    // QueueName
    let name_len = read_7bit_encoded_int(&mut cursor)?;
    let queue_name = {
        let mut name_buf = vec![0u8; name_len as usize];
        cursor.read_exact(&mut name_buf).ok()?;
        String::from_utf8(name_buf).ok()?
    };

    // QueueCapacity
    let mut queue_capacity_buf: [u8; 8] = [0; 8];
    cursor.read_exact(&mut queue_capacity_buf).ok()?;
    let queue_capacity = i64::from_le_bytes(queue_capacity_buf);

    // Validate
    if queue_capacity <= 0 {
        return None;
    }

    Some(ConnectionParams {
        queue_name,
        queue_capacity,
    })
}

/// Read a 7-bit encoded int in the format used by dotnet BinaryWriter/BinaryReader for
/// the length prefix on strings.
fn read_7bit_encoded_int(cursor: &mut Cursor<&[u8]>) -> Option<i32> {
    // Integer is encoded as a series of bytes, where the high bit
    // indicates whether there is another bytes to come, and the remaining
    // 7 bits are shifted into place from LSB to MSB. (Small numbers need fewer bytes this way.)

    const FLAG_MORE: u8 = 0x80;
    const MASK: u8 = !FLAG_MORE;

    let mut out: u32 = 0;
    let mut shift: u32 = 0;

    let mut buf: [u8; 1] = [0];

    // Keep grabbing bytes from the cursor until that fails or we've got our int.
    loop {
        cursor.read_exact(&mut buf).ok()?;

        out |= ((buf[0] & MASK) as u32) << shift;

        if buf[0] & FLAG_MORE == 0 {
            break;
        }

        shift += 7;
        if shift >= 32 {
            return None;
        }
    }

    Some(out as i32)
}

/// Scans `args` for the first complete `-QueueName` / `-QueueCapacity` pair (case-insensitive
/// flag suffix).
/// Requires QueueCapacity to be a positive integer.
/// Returns [`None`] if either flag is missing, malformed, or duplicated before
/// the pair completes.
fn parse_connection_args(args: &[impl AsRef<str>]) -> Option<ConnectionParams> {
    if args.is_empty() {
        return None;
    }

    let mut queue_name: Option<String> = None;
    let mut queue_capacity: Option<i64> = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let next_i = i + 1;
        if next_i >= args.len() {
            break;
        }

        let arg_lower = arg.as_ref().to_lowercase();
        if arg_lower.ends_with("queuename") {
            if queue_name.is_some() {
                return None;
            }
            queue_name = Some(args[next_i].as_ref().to_owned());
            i = next_i;
        } else if arg_lower.ends_with("queuecapacity") {
            if queue_capacity.is_some_and(|c| c > 0) {
                return None;
            }
            queue_capacity = args[next_i].as_ref().parse().ok().filter(|&c| c > 0);
            i = next_i;
        }

        i += 1;

        if let Some(name) = queue_name.as_ref()
            && let Some(cap) = queue_capacity
            && cap > 0
        {
            return Some(ConnectionParams {
                queue_name: name.clone(),
                queue_capacity: cap,
            });
        }
    }

    queue_name.and_then(|name| {
        queue_capacity
            .filter(|&c| c > 0)
            .map(|cap| ConnectionParams {
                queue_name: name,
                queue_capacity: cap,
            })
    })
}

/// Subscriber queue name for the renderer (non-authority -> `...A` side).
pub fn subscriber_queue_name(base: &str, channel: &str) -> String {
    format!("{base}{channel}A")
}

/// Publisher queue name for the renderer (non-authority -> `...S` side).
pub fn publisher_queue_name(base: &str, channel: &str) -> String {
    format!("{base}{channel}S")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_queue_name_and_capacity_case_insensitive() {
        let cmd = [
            "renderide",
            "-QueueName",
            "TestSession",
            "-QueueCapacity",
            "8388608",
        ];
        assert_eq!(
            parse_connection_args(&cmd),
            Some(ConnectionParams {
                queue_name: "TestSession".to_string(),
                queue_capacity: 8_388_608,
            })
        );
    }

    #[test]
    fn parse_args_accepts_queue_capacity_before_queue_name() {
        let cmd = [
            "renderide",
            "-QueueCapacity",
            "4096",
            "-QueueName",
            "LaterName",
        ];
        assert_eq!(
            parse_connection_args(&cmd),
            Some(ConnectionParams {
                queue_name: "LaterName".into(),
                queue_capacity: 4096,
            })
        );
    }

    #[test]
    fn parse_args_accepts_prefixed_flag_spellings_by_suffix() {
        let cmd = [
            "renderide",
            "--renderide-QueueName",
            "Prefixed",
            "/QueueCapacity",
            "2048",
        ];
        assert_eq!(
            parse_connection_args(&cmd),
            Some(ConnectionParams {
                queue_name: "Prefixed".into(),
                queue_capacity: 2048,
            })
        );
    }

    #[test]
    fn parse_args_returns_none_when_flag_value_is_missing() {
        assert_eq!(parse_connection_args(&["renderide", "-QueueName"]), None);
        assert_eq!(
            parse_connection_args(&["renderide", "-QueueName", "Name", "-QueueCapacity"]),
            None
        );
    }

    #[test]
    fn parse_args_allows_replacing_invalid_capacity_with_later_valid_capacity() {
        let cmd = [
            "renderide",
            "-QueueName",
            "Recover",
            "-QueueCapacity",
            "bad",
            "-QueueCapacity",
            "1024",
        ];

        assert_eq!(
            parse_connection_args(&cmd),
            Some(ConnectionParams {
                queue_name: "Recover".into(),
                queue_capacity: 1024,
            })
        );
    }

    #[test]
    fn parse_args_rejects_duplicate_queue_name() {
        let cmd = [
            "renderide",
            "-QueueName",
            "First",
            "-QueueName",
            "Second",
            "-QueueCapacity",
            "4096",
        ];
        assert_eq!(parse_connection_args(&cmd), None);
    }

    #[test]
    fn parse_args_returns_first_complete_pair_and_ignores_later_flags() {
        let cmd = [
            "renderide",
            "-QueueName",
            "S",
            "-QueueCapacity",
            "4096",
            "-QueueCapacity",
            "8192",
        ];
        assert_eq!(
            parse_connection_args(&cmd),
            Some(ConnectionParams {
                queue_name: "S".into(),
                queue_capacity: 4096,
            })
        );
    }

    #[test]
    fn parse_args_rejects_non_numeric_or_non_positive_capacity() {
        assert_eq!(
            parse_connection_args(&["r", "-QueueName", "n", "-QueueCapacity", "not_a_number"]),
            None
        );
        assert_eq!(
            parse_connection_args(&["r", "-QueueName", "n", "-QueueCapacity", "0"]),
            None
        );
        assert_eq!(
            parse_connection_args(&["r", "-QueueName", "n", "-QueueCapacity", "-100"]),
            None
        );
    }

    #[test]
    fn parse_args_returns_none_for_empty_argv() {
        assert_eq!(parse_connection_args(&Vec::<String>::new()), None);
    }

    #[test]
    fn ipc_suffixes_match_cloudtoid_non_authority() {
        let p = ConnectionParams {
            queue_name: "Foo".to_string(),
            queue_capacity: 1024,
        };
        assert_eq!(
            subscriber_queue_name(&p.queue_name, "Primary"),
            "FooPrimaryA"
        );
        assert_eq!(
            publisher_queue_name(&p.queue_name, "Primary"),
            "FooPrimaryS"
        );
        assert_eq!(
            subscriber_queue_name(&p.queue_name, "Background"),
            "FooBackgroundA"
        );
        assert_eq!(
            publisher_queue_name(&p.queue_name, "Background"),
            "FooBackgroundS"
        );
    }
}
