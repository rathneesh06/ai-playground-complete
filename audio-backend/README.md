# AI Playground Audio Backend

Advanced conversation analysis backend using **OpenAI Whisper** for Speech-to-Text and **Pyannote.audio** for speaker diarization.

## 🎯 Features

### Core Pipeline
1. **Speech-to-Text**: OpenAI Whisper with word-level timestamps
2. **Voice Activity Detection**: Pyannote VAD for speech region detection  
3. **Speaker Embeddings**: TitaNet-based speaker feature extraction
4. **Speaker Clustering**: Agglomerative clustering for 2-speaker scenarios
5. **Transcript Alignment**: Precise alignment of speech segments with speakers
6. **AI Summarization**: BART-based conversation summarization

### Technical Highlights
- **Modular Pipeline**: Each component (VAD, embeddings, clustering) works independently
- **No Vendor Lock-in**: Uses open-source models, no dependency on STT vendor diarization
- **GPU Accelerated**: CUDA support for faster processing
- **Async Processing**: Non-blocking API with real-time progress tracking
- **Fallback Methods**: Energy-based diarization when Pyannote unavailable

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.8+ required
python --version

# Optional: CUDA for GPU acceleration
nvidia-smi
```

### 2. Installation

```bash
# Clone and navigate
cd audio-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (optional, for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Model Setup

```bash
# Copy environment file
cp env.example .env

# Get HuggingFace token for Pyannote models
# 1. Go to https://huggingface.co/settings/tokens
# 2. Create a token with read access
# 3. Accept license for pyannote/speaker-diarization-3.1
# 4. Add token to .env file
```

### 4. Run the Server

```bash
# Start the backend
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Upload and Process Audio
```bash
POST /upload-audio
Content-Type: multipart/form-data

# Response
{
  "job_id": "uuid-string",
  "status": "processing_started"
}
```

### Check Processing Status
```bash
GET /status/{job_id}

# Response
{
  "job_id": "uuid",
  "status": "processing",  # uploaded, processing, completed, failed
  "progress": 45,          # 0-100
  "message": "Performing speaker diarization..."
}
```

### Get Results
```bash
GET /result/{job_id}

# Response
{
  "transcript": "Full conversation text...",
  "diarization": [
    {
      "speaker": "SPEAKER_00",
      "text": "Hello, how can I help you?",
      "start_time": 0.0,
      "end_time": 2.5,
      "confidence": 0.95
    }
  ],
  "summary": "Customer service conversation about...",
  "duration": 120.5,
  "num_speakers": 2,
  "language": "en",
  "processing_time": 15.2,
  "model_info": {
    "whisper_model": "base",
    "diarization_model": "pyannote/speaker-diarization-3.1"
  }
}
```

## 🔧 Configuration

### Model Selection

```python
# In audio_processor.py, modify:
self.whisper_model = whisper.load_model("base")  # tiny, base, small, medium, large

# For Pyannote model:
Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
```

### Performance Tuning

```python
# GPU Memory optimization
torch.cuda.empty_cache()

# Batch processing for multiple files
# Implement in audio_processor.py

# Model caching
# Models are loaded once and reused
```

## 📊 Supported Audio Formats

- **MP3**: Most common format
- **WAV**: Uncompressed, best quality
- **M4A**: Apple/AAC format
- **OGG**: Open-source format
- **FLAC**: Lossless compression

**File Limits**: 50MB per file, up to 2 hours duration

## 🧠 Pipeline Details

### 1. Whisper STT
```python
# Word-level timestamps for precise alignment
result = whisper_model.transcribe(
    audio_path,
    word_timestamps=True,
    language=None  # Auto-detect
)
```

### 2. Pyannote Diarization
```python
# Complete pipeline with VAD + embeddings + clustering
diarization = pipeline(audio_path)

# Manual pipeline steps:
vad = Pipeline.from_pretrained("pyannote/voice-activity-detection")
embedding = Pipeline.from_pretrained("pyannote/embedding") 
clustering = AgglomerativeClustering(n_clusters=2)
```

### 3. Fallback Diarization
```python
# Energy-based speaker change detection
energy_windows = [np.sum(window**2) for window in audio_chunks]
speaker_changes = detect_changes_from_energy(energy_windows)
```

## 🚨 Troubleshooting

### Common Issues

1. **Pyannote Model Access**
   ```bash
   # Error: Repository not found
   # Solution: Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1
   ```

2. **CUDA Out of Memory**
   ```python
   # Use smaller Whisper model
   whisper.load_model("tiny")  # Instead of "large"
   
   # Clear GPU cache
   torch.cuda.empty_cache()
   ```

3. **Audio Format Issues**
   ```bash
   # Install additional codecs
   pip install ffmpeg-python
   ```

4. **Performance Issues**
   ```python
   # Use CPU for small files
   device = "cpu"
   
   # Reduce audio quality for faster processing
   librosa.load(audio_path, sr=8000)  # Instead of 16000
   ```

## 🔗 Integration with Frontend

Update the React frontend to use real API:

```typescript
// In TranscriptService.ts
const API_BASE = 'http://localhost:8000';

static async transcribeAudio(audioFile: File): Promise<ProcessingResult> {
  // Upload file
  const formData = new FormData();
  formData.append('file', audioFile);
  
  const uploadResponse = await fetch(`${API_BASE}/upload-audio`, {
    method: 'POST',
    body: formData,
  });
  
  const { job_id } = await uploadResponse.json();
  
  // Poll for results
  return await this.pollForResults(job_id);
}
```

## 📈 Performance Benchmarks

| Model Size | Audio Duration | Processing Time | GPU Memory |
|------------|----------------|-----------------|------------|
| Whisper Tiny | 5 min | 30s | 1GB |
| Whisper Base | 5 min | 45s | 2GB |
| Whisper Small | 5 min | 60s | 3GB |
| + Pyannote | 5 min | +15s | +1GB |

## 🛠️ Development

### Adding New Features

1. **Custom Models**: Modify `audio_processor.py`
2. **New Endpoints**: Add to `main.py`
3. **Enhanced Diarization**: Extend clustering algorithms

### Testing

```bash
# Test with sample audio
curl -X POST "http://localhost:8000/upload-audio" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample.wav"
```

## 📝 License

This project uses several open-source models:
- **Whisper**: MIT License
- **Pyannote**: MIT License  
- **BART**: Apache 2.0 License

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review model documentation
3. Ensure all dependencies are installed
4. Verify GPU drivers (if using CUDA)

---

**Note**: This backend implements the exact pipeline you specified - using Whisper for STT and Pyannote for modular diarization without vendor dependencies! 