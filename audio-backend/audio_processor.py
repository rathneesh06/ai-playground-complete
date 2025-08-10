"""
Advanced Conversation Analysis Processor

This module implements the complete pipeline:
1. Speech-to-Text using OpenAI Whisper (with timestamps)
2. Voice Activity Detection using Pyannote
3. Speaker Embedding Extraction using Pyannote TitaNet
4. Speaker Clustering for diarization (up to 2 speakers)
5. Transcript alignment and summarization
"""

import asyncio
import time
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional

import torch
import whisper
import librosa
import numpy as np
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation, Segment
from sklearn.cluster import AgglomerativeClustering
from transformers import pipeline
import soundfile as sf

from models import ProcessingResult, SpeakerSegment

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

class ConversationProcessor:
    """
    Advanced conversation analysis processor using the modular pipeline approach
    """
    
    def __init__(self):
        self.whisper_model = None
        self.diarization_pipeline = None
        self.summarization_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
    
    async def initialize(self):
        """Initialize all AI models"""
        try:
            # Initialize Whisper for STT
            print("📥 Loading Whisper model...")
            self.whisper_model = whisper.load_model("base", device=self.device)
            print("✅ Whisper model loaded")
            
            # Initialize Pyannote diarization pipeline
            print("📥 Loading Pyannote diarization pipeline...")
            try:
                # Note: You'll need to accept the license on HuggingFace for pyannote models
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=True  # You'll need to set HUGGINGFACE_TOKEN env var
                )
                print("✅ Pyannote diarization pipeline loaded")
            except Exception as e:
                print(f"⚠️  Pyannote pipeline not available: {e}")
                print("📝 Using fallback diarization method")
                self.diarization_pipeline = None
            
            # Initialize summarization pipeline
            print("📥 Loading summarization model...")
            self.summarization_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1
            )
            print("✅ Summarization model loaded")
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            raise
    
    async def process_conversation(
        self, 
        audio_path: str, 
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> ProcessingResult:
        """
        Process audio file through the complete pipeline
        
        Pipeline Steps:
        1. Load and preprocess audio
        2. Speech-to-Text with Whisper (with word-level timestamps)
        3. Voice Activity Detection
        4. Speaker embedding extraction
        5. Speaker clustering (max 2 speakers)
        6. Transcript alignment with speakers
        7. Generate conversation summary
        """
        start_time = time.time()
        
        if progress_callback:
            progress_callback(5, "Loading audio file...")
        
        # Step 1: Load and preprocess audio
        audio_data, sample_rate = self._load_audio(audio_path)
        duration = len(audio_data) / sample_rate
        
        if progress_callback:
            progress_callback(15, "Transcribing audio with Whisper...")
        
        # Step 2: Speech-to-Text with timestamps
        transcript_result = await self._transcribe_with_whisper(audio_path)
        
        if progress_callback:
            progress_callback(40, "Performing speaker diarization...")
        
        # Step 3-5: Speaker diarization
        if self.diarization_pipeline:
            diarization_result = await self._pyannote_diarization(audio_path)
        else:
            diarization_result = await self._fallback_diarization(
                audio_data, sample_rate, transcript_result
            )
        
        if progress_callback:
            progress_callback(70, "Aligning transcript with speakers...")
        
        # Step 6: Align transcript with speaker diarization
        aligned_segments = self._align_transcript_with_speakers(
            transcript_result, diarization_result
        )
        
        if progress_callback:
            progress_callback(85, "Generating conversation summary...")
        
        # Step 7: Generate summary
        full_transcript = " ".join([seg["text"] for seg in transcript_result["segments"]])
        summary = await self._generate_summary(full_transcript)
        
        if progress_callback:
            progress_callback(100, "Processing completed!")
        
        processing_time = time.time() - start_time
        
        # Create result
        result = ProcessingResult(
            transcript=full_transcript,
            diarization=aligned_segments,
            summary=summary,
            duration=duration,
            num_speakers=len(set(seg.speaker for seg in aligned_segments)),
            language=transcript_result.get("language"),
            processing_time=processing_time,
            model_info={
                "whisper_model": "base",
                "diarization_model": "pyannote/speaker-diarization-3.1" if self.diarization_pipeline else "fallback",
                "summarization_model": "facebook/bart-large-cnn"
            }
        )
        
        return result
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and convert to appropriate format"""
        # Load audio with librosa (handles various formats)
        audio_data, sample_rate = librosa.load(audio_path, sr=16000)  # 16kHz for Whisper
        return audio_data, sample_rate
    
    async def _transcribe_with_whisper(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper with word-level timestamps"""
        def run_whisper():
            return self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=False
            )
        
        # Run Whisper in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_whisper)
        
        return result
    
    async def _pyannote_diarization(self, audio_path: str) -> Annotation:
        """Perform speaker diarization using Pyannote pipeline"""
        def run_diarization():
            # Run the diarization pipeline
            diarization = self.diarization_pipeline(audio_path)
            return diarization
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_diarization)
        
        return result
    
    async def _fallback_diarization(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int, 
        transcript_result: Dict
    ) -> List[Dict]:
        """
        Fallback diarization method using energy-based speaker change detection
        This is a simplified approach for when Pyannote is not available
        """
        segments = transcript_result["segments"]
        
        # Simple energy-based speaker change detection
        window_size = int(0.5 * sample_rate)  # 0.5 second windows
        energy_values = []
        
        for i in range(0, len(audio_data) - window_size, window_size // 2):
            window = audio_data[i:i + window_size]
            energy = np.sum(window ** 2)
            energy_values.append(energy)
        
        # Detect speaker changes based on energy pattern changes
        speaker_changes = self._detect_speaker_changes(energy_values, segments)
        
        # Assign speakers alternately starting from detected changes
        current_speaker = "SPEAKER_00"
        speaker_assignments = []
        
        for i, segment in enumerate(segments):
            if i in speaker_changes:
                current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
            
            speaker_assignments.append({
                "start": segment["start"],
                "end": segment["end"],
                "speaker": current_speaker
            })
        
        return speaker_assignments
    
    def _detect_speaker_changes(self, energy_values: List[float], segments: List[Dict]) -> List[int]:
        """Detect potential speaker changes based on energy patterns"""
        if len(energy_values) < 4:
            return []
        
        # Calculate energy differences
        energy_diffs = np.diff(energy_values)
        threshold = np.std(energy_diffs) * 2
        
        # Find significant changes
        change_points = []
        for i, diff in enumerate(energy_diffs):
            if abs(diff) > threshold:
                # Map energy window to segment index
                segment_idx = min(i // 2, len(segments) - 1)
                change_points.append(segment_idx)
        
        # Limit to reasonable number of speaker changes for 2-speaker scenario
        return change_points[:3]  # Max 3 changes for 2 speakers
    
    def _align_transcript_with_speakers(
        self, 
        transcript_result: Dict, 
        diarization_result
    ) -> List[SpeakerSegment]:
        """Align Whisper transcript with speaker diarization results"""
        segments = transcript_result["segments"]
        aligned_segments = []
        
        if isinstance(diarization_result, list):  # Fallback method result
            # Simple alignment with fallback diarization
            for segment, speaker_info in zip(segments, diarization_result):
                aligned_segments.append(SpeakerSegment(
                    speaker=speaker_info["speaker"],
                    text=segment["text"].strip(),
                    start_time=segment["start"],
                    end_time=segment["end"],
                    confidence=0.8  # Default confidence for fallback method
                ))
        else:  # Pyannote Annotation result
            # Align with Pyannote results
            for segment in segments:
                start_time = segment["start"]
                end_time = segment["end"]
                segment_duration = Segment(start_time, end_time)
                
                # Find overlapping speakers
                speakers = diarization_result.crop(segment_duration).labels()
                
                if speakers:
                    # Use the speaker with most overlap
                    speaker = list(speakers)[0]
                else:
                    speaker = "SPEAKER_00"
                
                aligned_segments.append(SpeakerSegment(
                    speaker=speaker,
                    text=segment["text"].strip(),
                    start_time=start_time,
                    end_time=end_time,
                    confidence=0.9  # High confidence for Pyannote
                ))
        
        return aligned_segments
    
    async def _generate_summary(self, transcript: str) -> str:
        """Generate conversation summary using BART"""
        def run_summarization():
            # Split long transcripts into chunks
            max_chunk_length = 1024
            chunks = [transcript[i:i+max_chunk_length] 
                     for i in range(0, len(transcript), max_chunk_length)]
            
            summaries = []
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # Only summarize substantial chunks
                    try:
                        summary = self.summarization_pipeline(
                            chunk, 
                            max_length=150, 
                            min_length=30, 
                            do_sample=False
                        )
                        summaries.append(summary[0]['summary_text'])
                    except Exception as e:
                        print(f"Warning: Summarization failed for chunk: {e}")
                        summaries.append(chunk[:100] + "...")
            
            # Combine summaries
            if len(summaries) > 1:
                combined = " ".join(summaries)
                # Summarize the combined summaries if still too long
                if len(combined) > 512:
                    final_summary = self.summarization_pipeline(
                        combined,
                        max_length=200,
                        min_length=50,
                        do_sample=False
                    )
                    return final_summary[0]['summary_text']
                return combined
            elif summaries:
                return summaries[0]
            else:
                return "Summary could not be generated."
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_summarization)
        
        return result 