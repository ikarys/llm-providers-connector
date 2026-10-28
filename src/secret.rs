//! Seals provider API keys at rest: the master key lives in a file of its
//! own, so the database alone (a backup, a copy) reveals no key.

use std::fmt;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use chacha20poly1305::ChaCha20Poly1305;
use chacha20poly1305::aead::{Aead, AeadCore, KeyInit, OsRng};

/// The key that seals API keys at rest: a stolen database alone reveals none.
pub struct MasterKey(chacha20poly1305::Key);

impl MasterKey {
    /// Reads the 32-byte key at `path`, creating it with mode 0600 on first use.
    pub fn load_or_create(path: &Path) -> Result<MasterKey, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("creating parent for {}: {e}", path.display()))?;
        }

        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
        {
            Ok(mut file) => {
                let key = ChaCha20Poly1305::generate_key(&mut OsRng);
                file.write_all(key.as_slice())
                    .map_err(|e| format!("writing {}: {e}", path.display()))?;
                Ok(MasterKey(key))
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => read_key(path),
            Err(e) => Err(format!("creating {}: {e}", path.display())),
        }
    }

    /// hex(nonce ‖ ciphertext), a fresh random 12-byte nonce each time.
    pub fn seal(&self, plain: &str) -> String {
        let cipher = ChaCha20Poly1305::new(&self.0);
        let nonce = ChaCha20Poly1305::generate_nonce(&mut OsRng);
        let ciphertext = cipher
            .encrypt(&nonce, plain.as_bytes())
            .expect("sealing in memory cannot fail");

        let mut bytes = Vec::with_capacity(nonce.len() + ciphertext.len());
        bytes.extend_from_slice(nonce.as_slice());
        bytes.extend_from_slice(&ciphertext);
        to_hex(&bytes)
    }

    pub fn open(&self, sealed: &str) -> Result<String, String> {
        let bytes = from_hex(sealed)?;
        if bytes.len() < 12 {
            return Err("sealed data must be at least 12 bytes".into());
        }

        let (nonce_bytes, ciphertext) = bytes.split_at(12);
        let nonce = chacha20poly1305::Nonce::from_slice(nonce_bytes);
        let cipher = ChaCha20Poly1305::new(&self.0);

        let plain = cipher.decrypt(nonce, ciphertext).map_err(|_| {
            "the key could not be unsealed: wrong master key or tampered data".to_string()
        })?;

        String::from_utf8(plain).map_err(|_| "unsealed key is not UTF-8".to_string())
    }
}

impl fmt::Debug for MasterKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MasterKey(..)")
    }
}

fn read_key(path: &Path) -> Result<MasterKey, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if bytes.len() != 32 {
        return Err(format!("{} must be exactly 32 bytes", path.display()));
    }

    let mut array = [0u8; 32];
    array.copy_from_slice(&bytes);
    Ok(MasterKey(chacha20poly1305::Key::from(array)))
}

const HEX: &[u8; 16] = b"0123456789abcdef";

fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

fn from_hex(input: &str) -> Result<Vec<u8>, String> {
    if !input.len().is_multiple_of(2) {
        return Err("bad hex".to_string());
    }

    let mut out = Vec::with_capacity(input.len() / 2);
    for pair in input.as_bytes().as_chunks::<2>().0 {
        let high = hex_digit(pair[0])?;
        let low = hex_digit(pair[1])?;
        out.push(high * 16 + low);
    }
    Ok(out)
}

fn hex_digit(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err("bad hex".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let counter = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "llm-providers-secret-{}-{}-{name}",
            std::process::id(),
            counter
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn seal_then_open_round_trips() {
        let dir = temp_dir("round-trip");
        let path = dir.join("master.key");
        let key = MasterKey::load_or_create(&path).unwrap();

        let sealed = key.seal("hello");
        assert_eq!(key.open(&sealed).unwrap(), "hello");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_seals_of_same_text_differ() {
        let dir = temp_dir("two-seals");
        let path = dir.join("master.key");
        let key = MasterKey::load_or_create(&path).unwrap();

        let a = key.seal("same");
        let b = key.seal("same");
        assert_ne!(a, b);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tampered_byte_fails_to_open() {
        let dir = temp_dir("tampered");
        let path = dir.join("master.key");
        let key = MasterKey::load_or_create(&path).unwrap();

        let sealed = key.seal("hello");
        let mut bytes = from_hex(&sealed).unwrap();
        bytes[20] ^= 0xff;

        let err = key.open(&to_hex(&bytes)).unwrap_err();
        assert_eq!(
            err,
            "the key could not be unsealed: wrong master key or tampered data"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn another_master_key_fails_to_open() {
        let dir = temp_dir("other-key");
        let path = dir.join("master.key");
        let other_path = dir.join("other.key");

        let key = MasterKey::load_or_create(&path).unwrap();
        let other = MasterKey::load_or_create(&other_path).unwrap();

        let sealed = key.seal("hello");
        let err = other.open(&sealed).unwrap_err();
        assert_eq!(
            err,
            "the key could not be unsealed: wrong master key or tampered data"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_or_create_twice_returns_same_key() {
        let dir = temp_dir("same-key");
        let path = dir.join("master.key");

        let first = MasterKey::load_or_create(&path).unwrap();
        let second = MasterKey::load_or_create(&path).unwrap();

        let sealed = first.seal("hello");
        assert_eq!(second.open(&sealed).unwrap(), "hello");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn created_file_has_mode_0600() {
        let dir = temp_dir("mode");
        let path = dir.join("master.key");

        let _ = MasterKey::load_or_create(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn wrong_length_file_is_error_and_left_untouched() {
        let dir = temp_dir("wrong-length");
        let path = dir.join("master.key");
        std::fs::write(&path, vec![0u8; 10]).unwrap();

        let err = MasterKey::load_or_create(&path).unwrap_err();
        assert!(err.contains(path.display().to_string().as_str()));
        assert!(err.contains("must be exactly 32 bytes"));

        assert_eq!(std::fs::read(&path).unwrap(), vec![0u8; 10]);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
