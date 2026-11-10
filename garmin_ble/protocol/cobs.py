import time
from collections import deque

class CobsCoDec:
    """
    Consistent Overhead Byte Stuffing (COBS) Encoder/Decoder for Garmin.
    """
    BUFFER_TIMEOUT_SEC = 1.5

    def __init__(self):
        self._buffer = bytearray()
        self._last_update = 0
        self._messages = deque()

    def received_bytes(self, data: bytes):
        now = time.time()
        if (now - self._last_update) > self.BUFFER_TIMEOUT_SEC:
            self._buffer.clear()
        self._last_update = now

        self._buffer.extend(data)
        self._decode()

    def reset(self):
        self._messages.clear()
        self._buffer.clear()

    def retrieve_message(self) -> bytes:
        """The next complete message, or ``None`` if there is none waiting.

        Call it until it returns ``None``: one notification can complete more
        than one frame, which happens routinely during a file transfer.
        """
        return self._messages.popleft() if self._messages else None

    def _decode(self):
        """Move every complete frame out of the buffer.

        A frame is ``00 <stuffed bytes, none of them zero> 00``. Frames can
        arrive split across notifications or several to a notification, so the
        buffer is cut at the zero delimiters rather than assumed to hold
        exactly one frame.
        """
        buffer = self._buffer
        while True:
            if len(buffer) < 4 or buffer[0] != 0x00:
                return
            end = buffer.find(0x00, 1)
            if end < 0:
                return
            if end < 3:
                # Too short to be a frame: the closing delimiter of one frame
                # followed directly by the opening delimiter of the next.
                del buffer[:end]
                continue

            frame = bytes(buffer[1:end])
            del buffer[: end + 1]
            self._messages.append(self._unstuff(frame))

    @staticmethod
    def _unstuff(frame: bytes) -> bytes:
        decoded = bytearray()
        i = 0
        while i < len(frame):
            code = frame[i]
            i += 1
            payload_size = code - 1
            decoded.extend(frame[i : i + payload_size])
            i += payload_size
            if code != 0xFF and i < len(frame):
                decoded.append(0x00)
        return bytes(decoded)

    @staticmethod
    def encode(data: bytes) -> bytes:
        encoded = bytearray()
        encoded.append(0x00)
        
        start_pos = 0
        last_byte_was_zero = False
        
        while start_pos < len(data):
            zero_index = start_pos
            while zero_index < len(data) and data[zero_index] != 0x00:
                zero_index += 1
                
            last_byte_was_zero = (zero_index < len(data))
            payload_size = zero_index - start_pos
            
            while payload_size >= 0xFE:
                encoded.append(0xFF)
                encoded.extend(data[start_pos : start_pos + 0xFE])
                payload_size -= 0xFE
                start_pos += 0xFE
                
            encoded.append(payload_size + 1)
            encoded.extend(data[start_pos : start_pos + payload_size])
            start_pos = zero_index + 1
            
        if last_byte_was_zero:
            encoded.append(0x01)
            
        encoded.append(0x00)
        return bytes(encoded)
