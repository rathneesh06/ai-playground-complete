"""
AI Playground Audio Backend
FastAPI server for conversation analysis with Whisper STT and Pyannote diarization
"""

import os
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from audio_processor import ConversationProcessor
from image_processor import ImageAnalysisProcessor
from document_processor import DocumentSummarizationProcessor
from models import ProcessingResult, ProcessingStatus, AnalysisRequest, ImageAnalysisResult, DocumentSummaryResult

# Initialize FastAPI app
app = FastAPI(
    title="AI Playground Audio Backend",
    description="Advanced conversation analysis with Whisper STT and Pyannote diarization",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.netlify.app", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor instances
conversation_processor = ConversationProcessor()
image_processor = ImageAnalysisProcessor()
document_processor = DocumentSummarizationProcessor()

# In-memory storage for processing status (use Redis in production)
processing_jobs: Dict[str, ProcessingStatus] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("🚀 Starting AI Playground Multi-Modal Backend...")
    print("📥 Loading AI models...")
    
    # Initialize all processors
    await conversation_processor.initialize()
    await image_processor.initialize()
    await document_processor.initialize()
    
    print("✅ Multi-modal backend ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Playground Audio Backend", "status": "running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "whisper_loaded": conversation_processor.whisper_model is not None,
        "pyannote_loaded": conversation_processor.diarization_pipeline is not None,
        "image_models_loaded": image_processor.primary_model is not None or image_processor.fallback_model is not None,
        "document_models_loaded": document_processor.primary_summarizer is not None or document_processor.fallback_summarizer is not None,
        "supported_formats": {
            "audio": ["mp3", "wav", "m4a", "ogg", "flac"],
            "image": ["jpg", "jpeg", "png", "gif", "bmp"],
            "document": ["pdf", "doc", "docx", "txt"],
            "url": ["http", "https"]
        },
        "max_file_size_mb": 50
    }

@app.post("/upload-audio")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> Dict[str, str]:
    """
    Upload audio file and start processing
    Returns a job ID for tracking progress
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    allowed_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (50MB limit)
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
    
    # Generate job ID
    import uuid
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    processing_jobs[job_id] = ProcessingStatus(
        job_id=job_id,
        status="uploaded",
        progress=0,
        message="File uploaded, starting processing..."
    )
    
    # Save file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    # Start background processing
    background_tasks.add_task(process_audio_background, job_id, temp_file_path, file.filename)
    
    return {"job_id": job_id, "status": "processing_started"}

@app.get("/status/{job_id}")
async def get_processing_status(job_id: str) -> ProcessingStatus:
    """Get processing status for a job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

@app.get("/result/{job_id}")
async def get_processing_result(job_id: str) -> ProcessingResult:
    """Get final processing result"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = processing_jobs[job_id]
    
    if status.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {status.status}"
        )
    
    if not status.result:
        raise HTTPException(status_code=500, detail="Result not available")
    
    return status.result

async def process_audio_background(job_id: str, file_path: str, original_filename: str):
    """Background task for processing audio"""
    try:
        # Update status
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].message = "Initializing audio processing..."
        
        # Process the audio
        result = await conversation_processor.process_conversation(
            file_path, 
            progress_callback=lambda progress, message: update_job_progress(job_id, progress, message)
        )
        
        # Update final status
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 100
        processing_jobs[job_id].message = "Processing completed successfully"
        processing_jobs[job_id].result = result
        
    except Exception as e:
        # Handle errors
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Processing failed: {str(e)}"
        print(f"❌ Processing failed for job {job_id}: {str(e)}")
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except:
            pass

def update_job_progress(job_id: str, progress: int, message: str):
    """Update job progress"""
    if job_id in processing_jobs:
        processing_jobs[job_id].progress = progress
        processing_jobs[job_id].message = message

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a processing job"""
    if job_id in processing_jobs:
        del processing_jobs[job_id]
        return {"message": "Job deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")

# Image Analysis Endpoints
@app.post("/upload-image")
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> Dict[str, str]:
    """
    Upload image file and start analysis
    Returns a job ID for tracking progress
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported image format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (50MB limit)
    content = await file.read()
    file_size = len(content)
    
    if file_size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
    
    # Generate job ID
    import uuid
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    processing_jobs[job_id] = ProcessingStatus(
        job_id=job_id,
        status="uploaded",
        progress=0,
        message="Image uploaded, starting analysis..."
    )
    
    # Save file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    # Start background processing
    background_tasks.add_task(process_image_background, job_id, temp_file_path, file.filename)
    
    return {"job_id": job_id, "status": "processing_started"}

# Document Summarization Endpoints
@app.post("/upload-document")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
) -> Dict[str, str]:
    """
    Upload document file and start summarization
    Returns a job ID for tracking progress
    """
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    allowed_extensions = {".pdf", ".doc", ".docx", ".txt"}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported document format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (50MB limit)
    content = await file.read()
    file_size = len(content)
    
    if file_size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(status_code=400, detail="File size exceeds 50MB limit")
    
    # Generate job ID
    import uuid
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    processing_jobs[job_id] = ProcessingStatus(
        job_id=job_id,
        status="uploaded",
        progress=0,
        message="Document uploaded, starting summarization..."
    )
    
    # Save file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    # Start background processing
    background_tasks.add_task(process_document_background, job_id, temp_file_path, file.filename)
    
    return {"job_id": job_id, "status": "processing_started"}

@app.post("/summarize-url")
async def summarize_url(
    background_tasks: BackgroundTasks,
    url: str
) -> Dict[str, str]:
    """
    Summarize content from URL
    Returns a job ID for tracking progress
    """
    
    # Validate URL
    from urllib.parse import urlparse
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValueError("Invalid URL")
        if result.scheme not in ['http', 'https']:
            raise ValueError("URL must be HTTP or HTTPS")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid URL provided")
    
    # Generate job ID
    import uuid
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    processing_jobs[job_id] = ProcessingStatus(
        job_id=job_id,
        status="uploaded",
        progress=0,
        message="URL provided, starting content extraction and summarization..."
    )
    
    # Start background processing
    background_tasks.add_task(process_url_background, job_id, url)
    
    return {"job_id": job_id, "status": "processing_started"}

# Background processing functions for new features
async def process_image_background(job_id: str, file_path: str, original_filename: str):
    """Background task for processing images"""
    try:
        # Update status
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].message = "Analyzing image with advanced MLLMs..."
        
        # Process the image
        result = await image_processor.analyze_image(
            file_path, 
            progress_callback=lambda progress, message: update_job_progress(job_id, progress, message)
        )
        
        # Update final status
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 100
        processing_jobs[job_id].message = "Image analysis completed successfully"
        processing_jobs[job_id].result = result
        
    except Exception as e:
        # Handle errors
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Image analysis failed: {str(e)}"
        print(f"❌ Image processing failed for job {job_id}: {str(e)}")
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except:
            pass

async def process_document_background(job_id: str, file_path: str, original_filename: str):
    """Background task for processing documents"""
    try:
        # Update status
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].message = "Extracting and summarizing document content..."
        
        # Process the document
        result = await document_processor.process_file(
            file_path, 
            progress_callback=lambda progress, message: update_job_progress(job_id, progress, message)
        )
        
        # Update final status
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 100
        processing_jobs[job_id].message = "Document summarization completed successfully"
        processing_jobs[job_id].result = result
        
    except Exception as e:
        # Handle errors
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Document processing failed: {str(e)}"
        print(f"❌ Document processing failed for job {job_id}: {str(e)}")
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except:
            pass

async def process_url_background(job_id: str, url: str):
    """Background task for processing URLs"""
    try:
        # Update status
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].message = "Extracting content from URL and summarizing..."
        
        # Process the URL
        result = await document_processor.process_url(
            url, 
            progress_callback=lambda progress, message: update_job_progress(job_id, progress, message)
        )
        
        # Update final status
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 100
        processing_jobs[job_id].message = "URL summarization completed successfully"
        processing_jobs[job_id].result = result
        
    except Exception as e:
        # Handle errors
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"URL processing failed: {str(e)}"
        print(f"❌ URL processing failed for job {job_id}: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    ) 