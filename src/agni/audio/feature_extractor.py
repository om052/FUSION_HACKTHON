"""Audio feature extraction for AGNI's audio-video synchronization pipeline.

Provides Syncnet-compatible MFCC feature extraction from 16 kHz mono PCM WAV
audio. This module is a pure feature-extraction stage: it does not perform
model inference, synchronization scoring, or deepfake classification.

Syncnet expects MFCC features computed with these exact parameters:
  - Sample rate:  16000 Hz
  - Channels:     mono (1)
  - Win length:   25 ms (400 samples at 16 kHz)
  - Win step:     10 ms (160 samples at 16 kHz)
  - Num cepstral: 13 coefficients
  - Num filters:  26 mel filterbanks
  - FFT size:     512 (next power of two >= window length; matches python_speech_features default)
  - Pre-emphasis: 0.97
  - Cepstral lifter: 22
  - Append energy: True (replaces 0th cepstral coefficient with log energy)
  - Window function: rectangular (default)

Audio input scaling (verified against SyncnetInstance.py):
  The original ``SyncNetInstance.evaluate`` method (joonson/syncnet_python)
  loads audio via ``scipy.io.wavfile.read`` — which returns raw int16 samples
  for ``pcm_s16le`` WAV files produced by ffmpeg — and passes those raw int16
  samples *directly* to ``python_speech_features.mfcc(audio, sample_rate)``
  without any normalization to [-1, 1].  Empirically, the log operation in the
  mel filterbank stage is NOT scale-invariant: passing int16 versus
  float32-normalized audio produces materially different MFCC values (max
  absolute difference ~20.8 on a test tone).  This module therefore preserves
  the raw int16 scaling to remain bit-for-bit compatible with the reference.

Output specification:
  - Shape: ``(T, 13)`` where ``T`` is the number of MFCC frames and 13 is the
    number of cepstral coefficients.
  - Orientation: each *row* is one MFCC frame; each *column* is one coefficient
    (coefficient 0 is log frame energy when ``appendEnergy=True``).
  - Dtype: ``numpy.float64`` (result of the log, DCT, and lifter pipeline in
    python_speech_features when operating on int16 input).
"""

from pathlib import Path
from typing import Union

import numpy as np
from python_speech_features import mfcc as _psf_mfcc
from scipy.io import wavfile

__all__ = [
    "AudioFeatureExtractor",
    "AudioFeatureExtractionError",
    "MalformedAudioInput",
    "SilentAudioInput",
    "InsufficientAudioDuration",
]

_WavPathLike = Union[str, Path]


class AudioFeatureExtractionError(Exception):
    """Base exception for all audio feature extraction failures."""


class MalformedAudioInput(AudioFeatureExtractionError):
    """Raised when a WAV file cannot be read or violates format constraints.

    Covers: file-not-found / unreadable WAV, unsupported sample rate,
    non-mono channel count, and unsupported PCM dtype.
    """


class SilentAudioInput(AudioFeatureExtractionError):
    """Raised when audio contains no audible signal.

    Covers both empty audio (zero samples) and digital silence
    (all samples are exactly zero).
    """


class InsufficientAudioDuration(AudioFeatureExtractionError):
    """Raised when audio is too short to produce the minimum downstream model window.

    The basic :meth:`AudioFeatureExtractor.extract_mfcc` does *not* enforce this;
    it is only checked in :meth:`AudioFeatureExtractor.extract_mfcc_for_syncnet`.
    """


class AudioFeatureExtractor:
    """Extracts Syncnet-compatible MFCC features from 16 kHz mono PCM WAV audio.

    All MFCC parameters are fixed to match the default configuration used by the
    original ``SyncnetInstance.py`` reference implementation (which calls
    ``python_speech_features.mfcc`` with positional ``signal`` and
    ``samplerate`` arguments only, relying on library defaults).

    Parameters:
        sample_rate: Target sample rate in Hz (must match WAV file). Default 16000.
        win_len: Analysis window length in seconds. Default 0.025.
        win_step: Step between successive windows in seconds. Default 0.010.
        num_cepstral: Number of cepstral coefficients. Default 13.
        num_filters: Number of mel filterbank bands. Default 26.
        n_fft: FFT size. Default 512 (next power of two ≥ window length;
            matches the python_speech_features default that Syncnet relies on).
        pre_emphasis: Pre-emphasis filter coefficient. Default 0.97.
        cep_lifter: Cepstral lifter parameter. Default 22.
        append_energy: If True, replace 0th cepstral coefficient with log energy.
            Default True.
        min_model_frames: Minimum number of MFCC frames required by the
            downstream Syncnet model window. Default 20.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        win_len: float = 0.025,
        win_step: float = 0.010,
        num_cepstral: int = 13,
        num_filters: int = 26,
        n_fft: int = 512,
        pre_emphasis: float = 0.97,
        cep_lifter: int = 22,
        append_energy: bool = True,
        min_model_frames: int = 20,
    ) -> None:
        self.sample_rate = sample_rate
        self.win_len = win_len
        self.win_step = win_step
        self.num_cepstral = num_cepstral
        self.num_filters = num_filters
        self.n_fft = n_fft
        self.pre_emphasis = pre_emphasis
        self.cep_lifter = cep_lifter
        self.append_energy = append_energy
        self.min_model_frames = min_model_frames

    # ------------------------------------------------------------------ #
    #  Shared validation
    # ------------------------------------------------------------------ #

    def _load_and_validate_wav(self, wav_path: _WavPathLike) -> np.ndarray:
        """Read a WAV file and validate its format constraints.

        Performs a single shared validation path used by all public extraction
        methods.  Does **not** perform RMS-based silence detection, voice
        activity detection, or speech recognition.

        Args:
            wav_path: Path to a WAV file (str or pathlib.Path).

        Returns:
            A 1-D ``int16`` numpy array of raw PCM samples.

        Raises:
            MalformedAudioInput: If the file is missing, unreadable, has an
                unsupported sample rate, is not mono, or uses an unsupported
                dtype.
            SilentAudioInput: If the audio is empty (zero samples) or contains
                only digital silence (all-zero samples).
        """
        path = Path(wav_path)

        if not path.exists():
            raise MalformedAudioInput(f"WAV file not found: {path}")

        try:
            sample_rate, audio = wavfile.read(str(path))
        except Exception as exc:
            raise MalformedAudioInput(
                f"Failed to read WAV file '{path}': {exc}"
            ) from exc

        # --- Sample rate ---
        if sample_rate != self.sample_rate:
            raise MalformedAudioInput(
                f"Unsupported sample rate: {sample_rate} Hz. "
                f"Expected {self.sample_rate} Hz."
            )

        # --- Channels: must be mono (1-D array) ---
        if audio.ndim != 1:
            raise MalformedAudioInput(
                f"Expected mono audio (1-D array), got a {audio.ndim}-D array "
                f"with shape {audio.shape}."
            )

        # --- Supported PCM dtype ---
        if audio.dtype != np.int16:
            raise MalformedAudioInput(
                f"Unsupported audio dtype: {audio.dtype}. "
                f"Expected int16 (pcm_s16le)."
            )

        # --- Empty audio (zero samples) ---
        if audio.size == 0:
            raise SilentAudioInput(
                "Audio contains no samples (empty WAV data segment)."
            )

        # --- Digital silence (all-zero samples) ---
        # NOTE: This is a simple all-zero check, NOT an RMS-based voice activity
        # detection or speech recognition pass.
        if not np.any(audio):
            raise SilentAudioInput(
                "Audio contains only digital silence (all samples are zero)."
            )

        return audio

    # ------------------------------------------------------------------ #
    #  Public extraction API
    # ------------------------------------------------------------------ #

    def extract_mfcc(self, wav_path: _WavPathLike) -> np.ndarray:
        """Extract basic MFCC features from a WAV file.

        Reads and validates the WAV, then computes MFCC features using
        ``python_speech_features.mfcc`` with parameters matching the original
        Syncnet reference.  Raw int16 samples are passed directly without
        normalization (verified source-matching behaviour).

        Args:
            wav_path: Path to a 16 kHz mono PCM (int16) WAV file.

        Returns:
            Float64 array of shape ``(T, 13)`` where ``T`` is the number of MFCC
            frames.  Each row is one frame; each column is one coefficient
            (0th column is log frame energy when ``append_energy`` is True).

        Raises:
            MalformedAudioInput: If the WAV fails validation.
            SilentAudioInput: If the audio is empty or silent.
        """
        audio = self._load_and_validate_wav(wav_path)

        features = _psf_mfcc(
            audio,
            samplerate=self.sample_rate,
            winlen=self.win_len,
            winstep=self.win_step,
            numcep=self.num_cepstral,
            nfilt=self.num_filters,
            nfft=self.n_fft,
            preemph=self.pre_emphasis,
            ceplifter=self.cep_lifter,
            appendEnergy=self.append_energy,
        )

        return features

    def extract_mfcc_for_syncnet(self, wav_path: _WavPathLike) -> np.ndarray:
        """Extract MFCC features with downstream minimum-frame validation.

        First calls :meth:`extract_mfcc` for full format validation and feature
        computation, then verifies that the resulting feature matrix contains at
        least ``min_model_frames`` rows.  The Syncnet model consumes 20-frame
        windows, so fewer than 20 frames cannot produce a valid model input.

        Args:
            wav_path: Path to a 16 kHz mono PCM (int16) WAV file.

        Returns:
            Float64 array of shape ``(T, 13)`` with ``T >= min_model_frames``.

        Raises:
            MalformedAudioInput: If the WAV fails validation.
            SilentAudioInput: If the audio is empty or silent.
            InsufficientAudioDuration: If fewer than ``min_model_frames`` MFCC
                frames are produced (audio too short for downstream model).
        """
        features = self.extract_mfcc(wav_path)

        if features.shape[0] < self.min_model_frames:
            raise InsufficientAudioDuration(
                f"Audio produced only {features.shape[0]} MFCC frames, "
                f"minimum required for downstream Syncnet model window is "
                f"{self.min_model_frames}."
            )

        return features
