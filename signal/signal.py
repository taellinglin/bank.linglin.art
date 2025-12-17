import wave
import struct
import json
import base64
import os
from typing import Tuple, Optional
import warnings
import numpy as np

class SerialNumberAudioEncoder:
    """
    Encodes and decodes serial numbers to/from WAV files using a frequency-based encoding.
    Similar to modem tones but optimized for accuracy with alphanumeric serial numbers.
    """
    
    # Encoding parameters
    SAMPLE_RATE = 44100  # CD quality
    BIT_DURATION = 0.04  # 40ms per bit (shorter for faster encoding)
    SILENCE_DURATION = 0.15  # 150ms silence between characters
    START_FREQ = 900  # Start tone frequency
    STOP_FREQ = 1000  # Stop tone frequency
    ZERO_FREQ = 600  # Frequency for 0
    ONE_FREQ = 1200   # Frequency for 1
    SYNC_FREQ = 800  # Sync tone
    
    # Character mapping for our specific serial number format
    CHAR_SET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"
    
    def __init__(self):
        self.char_to_bits = {char: format(idx, '06b') for idx, char in enumerate(self.CHAR_SET)}
        self.bits_to_char = {format(idx, '06b'): char for idx, char in enumerate(self.CHAR_SET)}
        
    def _generate_tone(self, frequency: float, duration: float, amplitude: float = 0.8) -> np.ndarray:
        """Generate a sine wave tone of given frequency and duration."""
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), False)
        tone = amplitude * np.sin(frequency * 2 * np.pi * t)
        
        # Add fade in/out to reduce clicking
        fade_samples = int(0.01 * self.SAMPLE_RATE)  # 10ms fade
        if fade_samples * 2 < len(tone):
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out
        
        return tone
    
    def _generate_silence(self, duration: float) -> np.ndarray:
        """Generate silence of given duration."""
        return np.zeros(int(self.SAMPLE_RATE * duration))
    
    def _calculate_checksum(self, data: str) -> int:
        """Calculate simple checksum for error detection."""
        return sum(ord(c) for c in data) % 256
    
    def encode_to_wav(self, serial_number: str, output_file: str = "serial_encoded.wav") -> str:
        """
        Encode a serial number to a WAV file.
        
        Format:
        1. Sync tone (SYNC_FREQ) for calibration
        2. Start tone (START_FREQ)
        3. Length indicator (8 bits)
        4. Each character (6 bits)
        5. Checksum (8 bits)
        6. Stop tone (STOP_FREQ)
        
        Returns the path to the created WAV file.
        """
        # Validate serial number
        for char in serial_number:
            if char not in self.CHAR_SET:
                raise ValueError(f"Invalid character '{char}' in serial number. Valid chars: {self.CHAR_SET}")
        
        print(f"Encoding serial number: {serial_number}")
        print(f"Length: {len(serial_number)} characters")
        
        # Initialize audio samples
        audio_samples = []
        
        # 1. Add calibration/sync tone (0.5 seconds)
        print("Adding sync tone...")
        audio_samples.append(self._generate_tone(self.SYNC_FREQ, 0.5))
        audio_samples.append(self._generate_silence(0.1))
        
        # 2. Add start tone (0.3 seconds)
        print("Adding start tone...")
        audio_samples.append(self._generate_tone(self.START_FREQ, 0.3))
        audio_samples.append(self._generate_silence(0.1))
        
        # 3. Encode length (8 bits)
        print("Encoding length...")
        length_bits = format(len(serial_number), '08b')
        print(f"Length bits: {length_bits} ({len(serial_number)})")
        
        for i, bit in enumerate(length_bits):
            freq = self.ONE_FREQ if bit == '1' else self.ZERO_FREQ
            audio_samples.append(self._generate_tone(freq, self.BIT_DURATION))
            if i < len(length_bits) - 1:  # No silence after last bit
                audio_samples.append(self._generate_silence(0.02))  # Small gap between bits
        
        audio_samples.append(self._generate_silence(self.SILENCE_DURATION))
        
        # 4. Encode each character
        print("Encoding characters...")
        for idx, char in enumerate(serial_number):
            # Get 6-bit representation
            bits = self.char_to_bits[char]
            print(f"  Char {idx+1}: '{char}' -> {bits}")
            
            # Encode bits
            for i, bit in enumerate(bits):
                freq = self.ONE_FREQ if bit == '1' else self.ZERO_FREQ
                audio_samples.append(self._generate_tone(freq, self.BIT_DURATION))
                if i < len(bits) - 1:  # No silence after last bit
                    audio_samples.append(self._generate_silence(0.02))
            
            # Add silence between characters (except after last character)
            if idx < len(serial_number) - 1:
                audio_samples.append(self._generate_silence(self.SILENCE_DURATION))
        
        # 5. Encode checksum (8 bits)
        checksum = self._calculate_checksum(serial_number)
        checksum_bits = format(checksum, '08b')
        print(f"Checksum: {checksum} -> {checksum_bits}")
        
        audio_samples.append(self._generate_silence(0.2))  # Pause before checksum
        
        for i, bit in enumerate(checksum_bits):
            freq = self.ONE_FREQ if bit == '1' else self.ZERO_FREQ
            audio_samples.append(self._generate_tone(freq, self.BIT_DURATION))
            if i < len(checksum_bits) - 1:
                audio_samples.append(self._generate_silence(0.02))
        
        audio_samples.append(self._generate_silence(0.2))
        
        # 6. Add stop tone
        print("Adding stop tone...")
        audio_samples.append(self._generate_tone(self.STOP_FREQ, 0.3))
        
        # Combine all samples
        audio = np.concatenate(audio_samples)
        
        # Normalize to 16-bit range
        audio = np.int16(audio * 32767 * 0.8)  # 80% volume to avoid clipping
        
        # Write to WAV file
        with wave.open(output_file, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(audio.tobytes())
        
        print(f"\nEncoding complete! File saved as: {output_file}")
        print(f"Audio duration: {len(audio) / self.SAMPLE_RATE:.2f} seconds")
        
        return output_file
    
    def _detect_frequency(self, audio_chunk: np.ndarray) -> float:
        """Detect the dominant frequency in an audio chunk."""
        # Apply window function to reduce spectral leakage
        window = np.hanning(len(audio_chunk))
        windowed_audio = audio_chunk * window
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed_audio)
        freqs = np.fft.rfftfreq(len(windowed_audio), 1/self.SAMPLE_RATE)
        magnitudes = np.abs(fft_result)
        
        # Find peak in expected frequency ranges
        valid_freqs = [self.ZERO_FREQ, self.ONE_FREQ, self.START_FREQ, self.STOP_FREQ, self.SYNC_FREQ]
        
        # Look for peaks near our expected frequencies
        detected_freq = 0
        max_magnitude = 0
        
        for target_freq in valid_freqs:
            # Find indices near target frequency (±100 Hz)
            freq_range = (freqs > target_freq - 150) & (freqs < target_freq + 150)
            if np.any(freq_range):
                local_max_idx = np.argmax(magnitudes[freq_range])
                local_max_freq = freqs[freq_range][local_max_idx]
                local_max_mag = magnitudes[freq_range][local_max_idx]
                
                if local_max_mag > max_magnitude:
                    max_magnitude = local_max_mag
                    detected_freq = local_max_freq
        
        return detected_freq
    
    def _find_tones(self, audio: np.ndarray, threshold: float = 0.05):
        """Find tone positions in audio."""
        # Energy envelope
        envelope = np.abs(audio)
        
        # Smooth the envelope
        window_size = int(0.01 * self.SAMPLE_RATE)  # 10ms window
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(envelope, kernel, mode='same')
        
        # Find where audio exceeds threshold
        tone_starts = []
        tone_ends = []
        in_tone = False
        
        for i in range(len(smoothed)):
            if not in_tone and smoothed[i] > threshold:
                tone_starts.append(i)
                in_tone = True
            elif in_tone and smoothed[i] <= threshold * 0.5:
                tone_ends.append(i)
                in_tone = False
        
        if in_tone:
            tone_ends.append(len(smoothed) - 1)
        
        return list(zip(tone_starts, tone_ends))
    
    def decode_from_wav(self, input_file: str) -> Tuple[str, bool]:
        """
        Decode a serial number from a WAV file.
        
        Returns:
            Tuple of (decoded_serial_number, success_flag)
        """
        print(f"\nDecoding from: {input_file}")
        
        # Read WAV file
        with wave.open(input_file, 'r') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        
        print(f"Audio loaded: {len(audio)} samples ({len(audio)/self.SAMPLE_RATE:.2f} seconds)")
        
        # Find all tones in the audio
        tones = self._find_tones(audio, threshold=0.03)
        print(f"Found {len(tones)} tone segments")
        
        # Skip the sync tone (first long tone)
        if len(tones) > 0 and (tones[0][1] - tones[0][0]) / self.SAMPLE_RATE > 0.4:
            tones = tones[1:]  # Remove sync tone
        
        # Skip start tone
        if len(tones) > 0:
            # Check if first tone is start frequency
            start_chunk = audio[tones[0][0]:tones[0][1]]
            start_freq = self._detect_frequency(start_chunk)
            if abs(start_freq - self.START_FREQ) < 200:
                tones = tones[1:]  # Remove start tone
                print("Start tone detected and skipped")
        
        # Decode bits from remaining tones
        bits = []
        for start, end in tones:
            # Skip very short tones (likely noise)
            duration = (end - start) / self.SAMPLE_RATE
            if duration < self.BIT_DURATION * 0.5:
                continue
            
            chunk = audio[start:end]
            freq = self._detect_frequency(chunk)
            
            # Classify as 0, 1, or stop
            if abs(freq - self.ZERO_FREQ) < 150:
                bits.append('0')
            elif abs(freq - self.ONE_FREQ) < 150:
                bits.append('1')
            elif abs(freq - self.STOP_FREQ) < 200:
                print("Stop tone detected")
                break
        
        bit_string = ''.join(bits)
        print(f"Decoded bit string ({len(bit_string)} bits): {bit_string}")
        
        # Parse the bits
        if len(bit_string) < 8:
            print("Error: Not enough bits decoded")
            return "", False
        
        # First 8 bits are length
        length_bits = bit_string[:8]
        expected_length = int(length_bits, 2)
        print(f"Expected length: {expected_length} (bits: {length_bits})")
        
        # Remaining bits (excluding checksum)
        data_bits = bit_string[8:]
        
        # Each character is 6 bits
        if len(data_bits) < expected_length * 6 + 8:  # +8 for checksum
            print(f"Error: Not enough data bits. Got {len(data_bits)}, need at least {expected_length * 6 + 8}")
            return "", False
        
        # Extract character bits
        decoded_chars = []
        for i in range(expected_length):
            start_idx = i * 6
            end_idx = start_idx + 6
            if end_idx > len(data_bits):
                break
            
            char_bits = data_bits[start_idx:end_idx]
            
            if char_bits in self.bits_to_char:
                char = self.bits_to_char[char_bits]
                decoded_chars.append(char)
            else:
                print(f"Warning: Invalid bit pattern: {char_bits}")
                decoded_chars.append('?')
        
        serial_number = ''.join(decoded_chars)
        
        # Extract checksum (8 bits after the characters)
        checksum_start = expected_length * 6
        checksum_end = checksum_start + 8
        if checksum_end <= len(data_bits):
            checksum_bits = data_bits[checksum_start:checksum_end]
            received_checksum = int(checksum_bits, 2)
            calculated_checksum = self._calculate_checksum(serial_number)
            
            print(f"Received checksum: {received_checksum} (bits: {checksum_bits})")
            print(f"Calculated checksum: {calculated_checksum}")
            
            if received_checksum != calculated_checksum:
                print("Warning: Checksum mismatch!")
                # Still return the serial number but mark as potentially corrupted
                return serial_number, False
        
        print(f"Successfully decoded: {serial_number}")
        return serial_number, True
    
    def create_test_file(self, output_file: str = "test_serial.wav") -> str:
        """Create a test WAV file with a known serial number."""
        test_sn = "SN-XNs98z0GTTR5Lg-W6Q"
        return self.encode_to_wav(test_sn, output_file)


def test_encoding_decoding():
    """Test the encoder/decoder with the provided serial number."""
    encoder = SerialNumberAudioEncoder()
    
    # Your serial number
    serial_number = "SN-XNs98z0GTTR5Lg-W6Q"
    
    print("=" * 60)
    print("Serial Number Audio Encoder/Decoder")
    print("=" * 60)
    print(f"\nOriginal Serial Number: {serial_number}")
    
    # Encode to WAV
    print("\n" + "=" * 60)
    print("ENCODING")
    print("=" * 60)
    
    try:
        wav_file = encoder.encode_to_wav(serial_number, "serial_encoded.wav")
        
        # Decode from WAV
        print("\n" + "=" * 60)
        print("DECODING")
        print("=" * 60)
        
        decoded, success = encoder.decode_from_wav(wav_file)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Original:  {serial_number}")
        print(f"Decoded:   {decoded}")
        print(f"Match:     {serial_number == decoded}")
        print(f"Success:   {success}")
        
        # Create a test file
        print("\n" + "=" * 60)
        print("CREATING TEST FILE")
        print("=" * 60)
        test_file = encoder.create_test_file("test_output.wav")
        print(f"Test file created: {test_file}")
        
        return serial_number == decoded and success
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_encode_decode():
    """Simple function for quick encoding and decoding."""
    encoder = SerialNumberAudioEncoder()
    sn = "SN-XNs98z0GTTR5Lg-W6Q"
    
    print("Quick encode/decode test:")
    print(f"Serial: {sn}")
    
    # Encode
    encoded_file = encoder.encode_to_wav(sn, "quick_test.wav")
    print(f"Encoded to: {encoded_file}")
    
    # Decode
    decoded, success = encoder.decode_from_wav(encoded_file)
    print(f"Decoded: {decoded}")
    print(f"Success: {success}")
    print(f"Match: {sn == decoded}")
    
    return sn == decoded


if __name__ == "__main__":
    # Run comprehensive test
    print("Starting Serial Number Audio Encoder/Decoder...\n")
    
    success = test_encoding_decoding()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ TEST PASSED! Encoding/decoding successful.")
    else:
        print("✗ TEST FAILED! Check the output above for errors.")
    print("=" * 60)
    
    # Optional: Also run quick test
    # quick_encode_decode()