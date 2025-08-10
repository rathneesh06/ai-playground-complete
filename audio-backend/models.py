"""
Data models for the audio processing API
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class SpeakerSegment(BaseModel):
    """Individual speaker segment with timing information"""
    speaker: str
    text: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None

class ProcessingResult(BaseModel):
    """Complete conversation analysis result"""
    # Raw transcript with timestamps
    transcript: str
    
    # Speaker diarization results
    diarization: List[SpeakerSegment]
    
    # AI-generated summary
    summary: str
    
    # Metadata
    duration: float  # in seconds
    num_speakers: int
    language: Optional[str] = None
    
    # Processing statistics
    processing_time: float
    model_info: Dict[str, str]

class ProcessingStatus(BaseModel):
    """Status of an audio processing job"""
    job_id: str
    status: str  # uploaded, processing, completed, failed
    progress: int  # 0-100
    message: str
    
    # Optional result when completed
    result: Optional[ProcessingResult] = None
    
    # Timestamps
    created_at: datetime = None
    updated_at: datetime = None
    
    def __init__(self, **data):
        if data.get('created_at') is None:
            data['created_at'] = datetime.now()
        if data.get('updated_at') is None:
            data['updated_at'] = datetime.now()
        super().__init__(**data)

class AnalysisRequest(BaseModel):
    """Request for conversation analysis"""
    file_path: str
    options: Optional[Dict[str, Any]] = {}

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    job_id: Optional[str] = None

class ImageAnalysisResult(BaseModel):
    """Image analysis result with hyper-detailed captions"""
    primary_caption: str
    detailed_description: str
    hyper_detailed_caption: str
    objects_detected: List[Dict[str, Any]]
    spatial_relationships: List[str]
    context_analysis: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_time: float
    model_info: Dict[str, str]

class DocumentSummaryResult(BaseModel):
    """Document summarization result with multi-level summaries"""
    original_length: int
    summary_short: str
    summary_medium: str
    summary_long: str
    key_points: List[str]
    document_type: str
    language: str
    confidence_score: float
    compression_ratio: float
    processing_time: float
    model_info: Dict[str, str]

class HealthStatus(BaseModel):
    """Health check response"""
    status: str
    whisper_loaded: bool
    pyannote_loaded: bool
    image_models_loaded: bool
    document_models_loaded: bool
    supported_formats: List[str]
    max_file_size_mb: int
    uptime: Optional[float] = None 