# 🤖 AI Playground - Multi-Modal AI Experience

A complete full-stack application featuring three cutting-edge AI capabilities: Conversation Analysis, Image Analysis, and Document Summarization.

## 🌟 Features

### 🎤 Conversation Analysis
- **Speech-to-Text**: OpenAI Whisper with word-level timestamps
- **Speaker Diarization**: Custom Pyannote.audio pipeline (no vendor dependency)
- **Smart Clustering**: 2-speaker identification and segmentation
- **AI Summarization**: Intelligent conversation summaries

### 🖼️ Image Analysis
- **Advanced MLLMs**: InternVL3-style multimodal processing
- **Object Detection**: DETR-ResNet-50 with spatial analysis
- **Hyper-detailed Captions**: IIW framework-inspired descriptions
- **Context Analysis**: Scene type, complexity, and relationship detection

### 📄 Document Summarization
- **Hybrid Pipeline**: Extractive-Abstractive summarization
- **SOTA Models**: PEGASUS + BART for optimal results
- **Multi-format Support**: PDF, DOC, DOCX, TXT files
- **URL Processing**: Web content extraction and summarization
- **Multi-level Summaries**: Short, Medium, and Long versions

## 🏗️ Architecture

### Frontend (React + TypeScript)
- **Framework**: React 18 with TypeScript
- **UI Library**: Material-UI (MUI) for professional design
- **State Management**: React Context API
- **Routing**: React Router v6
- **Real-time Updates**: Progress tracking for AI processing

### Backend (Python + FastAPI)
- **Framework**: FastAPI with async processing
- **AI Models**: Whisper, Pyannote, BLIP-2, PEGASUS, BART
- **Processing**: Asynchronous job management
- **APIs**: RESTful endpoints with real-time status updates

## 🚀 Live Demo

- **Frontend**: [https://ai-playground-full.vercel.app](https://ai-playground-full.vercel.app)
- **Backend**: Deployed on Render with full AI processing

## 🛠️ Tech Stack

### Frontend
- React 18, TypeScript, Material-UI
- Vercel deployment with global CDN
- Responsive design, PWA-ready

### Backend
- Python 3.9, FastAPI, Uvicorn
- OpenAI Whisper, Pyannote.audio, Transformers
- CUDA support for GPU acceleration

### AI Models
- **STT**: OpenAI Whisper (multilingual)
- **Diarization**: Pyannote.audio with TitaNet embeddings
- **Vision**: BLIP-2 OPT-6.7B for image captioning
- **Summarization**: PEGASUS-XSUM, BART-Large-CNN

## 📦 Installation

### Frontend
```bash
cd plivo-playground
npm install
npm start
```

### Backend
```bash
cd audio-backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## 🌐 Deployment

### Frontend (Vercel)
```bash
npm run build
vercel --prod
```

### Backend (Render/Railway)
- Connect GitHub repository
- Set build command: `pip install -r requirements.txt`
- Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## 🎯 Usage

1. **Login** with any credentials (demo authentication)
2. **Select AI Skill** from dropdown menu
3. **Upload Content** (audio, images, or documents)
4. **Get AI-Powered Results** with detailed analysis

## 🏆 Key Achievements

- ✅ **Multi-modal AI Integration**: Three different AI domains
- ✅ **SOTA Model Implementation**: Latest research models
- ✅ **Production Architecture**: Scalable, async, error-handling
- ✅ **Professional UI/UX**: Material Design principles
- ✅ **Full-Stack Deployment**: Global CDN + AI backend

## 📈 Performance

- **Global CDN**: Sub-second loading times worldwide
- **Async Processing**: Non-blocking AI operations
- **Progress Tracking**: Real-time status updates
- **Error Handling**: Graceful degradation and fallbacks

## 🤝 Contributing

This project demonstrates modern full-stack development with cutting-edge AI integration. Feel free to explore the codebase and suggest improvements!

## 📄 License

MIT License - feel free to use for learning and development.

---

**Built with ❤️ using the latest AI technologies and modern web development practices.**
