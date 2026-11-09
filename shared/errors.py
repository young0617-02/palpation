"""Error codes aligned with the scheme §2.7 (mainline reuse + edge extras)."""

E_TIMEOUT = "E_TIMEOUT"
E_NOT_FOUND = "E_NOT_FOUND"
E_INVALID_ARG = "E_INVALID_ARG"
E_UPSTREAM = "E_UPSTREAM"
E_CONFLICT = "E_CONFLICT"
E_PERMISSION = "E_PERMISSION"
E_MODEL_HASH_MISMATCH = "E_MODEL_HASH_MISMATCH"
E_REQUIRES_FIRMWARE = "E_REQUIRES_FIRMWARE"
E_DISK_FULL = "E_DISK_FULL"
E_UNSIGNED_MODEL = "E_UNSIGNED_MODEL"


class EdgeError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
