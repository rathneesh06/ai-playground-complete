#!/bin/bash

# AI Playground Audio Backend Setup Script
# This script sets up the complete environment for the conversation analysis backend

set -e  # Exit on any error

echo "🚀 AI Playground Audio Backend Setup"
echo "====================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.8+ is installed
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    print_success "Dependencies installed"
}

# Setup environment file
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        cp env.example .env
        print_success "Environment file created from template"
        print_warning "Please edit .env file with your HuggingFace token"
        print_warning "Get token from: https://huggingface.co/settings/tokens"
        print_warning "Accept license for: https://huggingface.co/pyannote/speaker-diarization-3.1"
    else
        print_warning "Environment file already exists"
    fi
}

# Check GPU availability
check_gpu() {
    print_status "Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        
        print_status "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        print_success "CUDA-enabled PyTorch installed"
    else
        print_warning "No NVIDIA GPU detected, using CPU-only PyTorch"
    fi
}

# Download and cache models
download_models() {
    print_status "Pre-downloading AI models (this may take a while)..."
    
    # Create a simple Python script to download models
    cat > download_models.py << 'EOF'
import warnings
warnings.filterwarnings("ignore")

import whisper
import torch
from transformers import pipeline

print("📥 Downloading Whisper model...")
try:
    model = whisper.load_model("base")
    print("✅ Whisper model downloaded")
except Exception as e:
    print(f"❌ Error downloading Whisper: {e}")

print("📥 Downloading summarization model...")
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("✅ Summarization model downloaded")
except Exception as e:
    print(f"❌ Error downloading BART: {e}")

print("📥 Checking Pyannote access...")
try:
    from pyannote.audio import Pipeline
    # This will fail if token is not set, but that's expected
    print("✅ Pyannote library available (token setup required)")
except Exception as e:
    print(f"⚠️  Pyannote setup needed: {e}")

print("🎉 Model downloads completed!")
EOF

    python download_models.py
    rm download_models.py
    
    print_success "Model downloads completed"
}

# Create startup script
create_startup_script() {
    print_status "Creating startup script..."
    
    cat > start_backend.sh << 'EOF'
#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Start the backend server
echo "🚀 Starting AI Playground Audio Backend..."
echo "📡 Server will be available at: http://localhost:8000"
echo "📋 API docs available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py
EOF

    chmod +x start_backend.sh
    print_success "Startup script created: ./start_backend.sh"
}

# Main setup function
main() {
    echo ""
    print_status "Starting setup process..."
    echo ""
    
    # Check prerequisites
    check_python
    
    # Setup environment
    create_venv
    install_dependencies
    setup_env
    check_gpu
    
    # Download models (optional)
    read -p "Download AI models now? (recommended, ~2GB download) [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_models
    else
        print_warning "Models will be downloaded on first use"
    fi
    
    # Create helper scripts
    create_startup_script
    
    echo ""
    print_success "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file with your HuggingFace token"
    echo "2. Accept license: https://huggingface.co/pyannote/speaker-diarization-3.1"
    echo "3. Start the backend: ./start_backend.sh"
    echo "4. Test at: http://localhost:8000/health"
    echo ""
    print_status "For detailed instructions, see README.md"
}

# Run main function
main 