import React, { useCallback, useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Alert,
} from '@mui/material';
import { CloudUpload, Description } from '@mui/icons-material';

interface DocumentUploadProps {
  onFileSelect: (file: File) => void;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({ onFileSelect }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateFile = useCallback((file: File): boolean => {
    const validTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
    const maxSize = 50 * 1024 * 1024; // 50MB

    const fileExtension = file.name.toLowerCase();
    const isValidExtension = fileExtension.endsWith('.pdf') || 
                           fileExtension.endsWith('.doc') || 
                           fileExtension.endsWith('.docx') || 
                           fileExtension.endsWith('.txt');

    if (!validTypes.includes(file.type) && !isValidExtension) {
      setError('Please upload a valid document file (PDF, DOC, DOCX, TXT)');
      return false;
    }

    if (file.size > maxSize) {
      setError('File size must be less than 50MB');
      return false;
    }

    setError(null);
    return true;
  }, []);

  const handleFile = useCallback((file: File) => {
    if (validateFile(file)) {
      onFileSelect(file);
    }
  }, [onFileSelect, validateFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFile(files[0]);
    }
  }, [handleFile]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  return (
    <Box>
      <Paper
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        sx={{
          p: 4,
          textAlign: 'center',
          border: 2,
          borderStyle: 'dashed',
          borderColor: isDragOver ? 'secondary.main' : 'grey.300',
          bgcolor: isDragOver ? 'secondary.50' : 'grey.50',
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          '&:hover': {
            borderColor: 'secondary.main',
            bgcolor: 'secondary.50',
          },
        }}
      >
        <Description sx={{ fontSize: 60, color: 'secondary.main', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          Upload Document for Summarization
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Drag and drop a document here, or click to browse
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 3, display: 'block' }}>
          Supported formats: PDF, DOC, DOCX, TXT (Max: 50MB)
        </Typography>
        
        <input
          type="file"
          accept=".pdf,.doc,.docx,.txt"
          onChange={handleFileInput}
          style={{ display: 'none' }}
          id="document-upload"
        />
        <label htmlFor="document-upload">
          <Button
            variant="outlined"
            component="span"
            startIcon={<CloudUpload />}
            size="large"
            color="secondary"
          >
            Browse Documents
          </Button>
        </label>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}
    </Box>
  );
}; 