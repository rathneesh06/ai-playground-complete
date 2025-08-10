import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  LinearProgress,
  Alert,
  Tabs,
  Tab,
  Chip,
} from '@mui/material';
import { AudioFile, Person, Summarize } from '@mui/icons-material';
import { AudioUpload } from './AudioUpload';
import { TranscriptService } from '../../services/TranscriptService';
import { DiarizationService } from '../../services/DiarizationService';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

interface AnalysisResult {
  transcript: string;
  diarization: Array<{
    speaker: string;
    text: string;
    timestamp: string;
  }>;
  summary: string;
}

const ConversationAnalysis: React.FC = () => {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(0);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'available' | 'unavailable'>('checking');

  const handleFileSelect = (file: File) => {
    setAudioFile(file);
    setResult(null);
    setError(null);
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  // Check backend status on component mount
  React.useEffect(() => {
    const checkBackend = async () => {
      try {
        const isAvailable = await TranscriptService.isBackendAvailable();
        setBackendStatus(isAvailable ? 'available' : 'unavailable');
      } catch (error) {
        setBackendStatus('unavailable');
      }
    };
    
    checkBackend();
  }, []);

  const processAudio = async () => {
    if (!audioFile) return;

    setLoading(true);
    setProgress(0);
    setError(null);

    try {
      // Check if backend is available
      const backendAvailable = await TranscriptService.isBackendAvailable();
      
      if (backendAvailable) {
        // Use real backend API
        setProgress(10);
        const analysisResult = await TranscriptService.analyzeConversation(audioFile);
        
        setProgress(100);
        setResult({
          transcript: analysisResult.transcript,
          diarization: analysisResult.diarization,
          summary: analysisResult.summary,
        });
      } else {
        // Fallback to mock processing for demo
        setProgress(30);
        const transcript = await TranscriptService.mockTranscribeAudio();
        
        setProgress(70);
        const diarization = await DiarizationService.diarizeTranscript(transcript);
        
        setProgress(90);
        const summary = await TranscriptService.generateSummary(transcript);
        
        setProgress(100);
        setResult({
          transcript,
          diarization,
          summary,
        });
      }
      
      setActiveTab(0); // Show transcript tab first
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during processing');
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <Typography variant="h5" fontWeight="bold">
          Conversation Analysis
        </Typography>
        <Chip
          label={
            backendStatus === 'checking' ? 'Checking Backend...' :
            backendStatus === 'available' ? 'AI Backend Active' : 'Demo Mode'
          }
          color={
            backendStatus === 'checking' ? 'default' :
            backendStatus === 'available' ? 'success' : 'warning'
          }
          size="small"
        />
      </Box>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload an audio file to get transcript, speaker diarization, and summary
        {backendStatus === 'unavailable' && (
          <Typography variant="body2" color="warning.main" sx={{ mt: 1 }}>
            ⚠️ Backend not available - using demo data. Start the Python backend for real processing.
          </Typography>
        )}
      </Typography>

      {/* Audio Upload Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <AudioUpload onFileSelect={handleFileSelect} />
        
        {audioFile && (
          <Box sx={{ mt: 2 }}>
            <Alert severity="info" sx={{ mb: 2 }}>
              <Typography variant="body2">
                Selected: {audioFile.name} ({(audioFile.size / 1024 / 1024).toFixed(2)} MB)
              </Typography>
            </Alert>
            
            <Button
              variant="contained"
              onClick={processAudio}
              disabled={loading}
              startIcon={<AudioFile />}
              size="large"
            >
              {loading ? 'Processing...' : 'Analyze Conversation'}
            </Button>
          </Box>
        )}

        {loading && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" sx={{ mb: 1 }}>
              Processing audio... {progress}%
            </Typography>
            <LinearProgress variant="determinate" value={progress} />
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </Paper>

      {/* Results Section */}
      {result && (
        <Paper sx={{ mt: 3 }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={activeTab} onChange={handleTabChange}>
              <Tab
                label="Transcript"
                icon={<AudioFile />}
                iconPosition="start"
              />
              <Tab
                label="Speaker Diarization"
                icon={<Person />}
                iconPosition="start"
              />
              <Tab
                label="Summary"
                icon={<Summarize />}
                iconPosition="start"
              />
            </Tabs>
          </Box>

          <TabPanel value={activeTab} index={0}>
            <Typography variant="h6" gutterBottom>
              Full Transcript
            </Typography>
            <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                {result.transcript}
              </Typography>
            </Paper>
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            <Typography variant="h6" gutterBottom>
              Speaker Diarization
            </Typography>
            <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
              {result.diarization.map((segment, index) => (
                <Paper
                  key={index}
                  variant="outlined"
                  sx={{ p: 2, mb: 2, bgcolor: segment.speaker === 'Speaker 1' ? 'blue.50' : 'green.50' }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Chip
                      label={segment.speaker}
                      color={segment.speaker === 'Speaker 1' ? 'primary' : 'secondary'}
                      size="small"
                      sx={{ mr: 1 }}
                    />
                    <Typography variant="caption" color="text.secondary">
                      {segment.timestamp}
                    </Typography>
                  </Box>
                  <Typography variant="body1">
                    {segment.text}
                  </Typography>
                </Paper>
              ))}
            </Box>
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            <Typography variant="h6" gutterBottom>
              Conversation Summary
            </Typography>
            <Paper variant="outlined" sx={{ p: 2, bgcolor: 'grey.50' }}>
              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                {result.summary}
              </Typography>
            </Paper>
          </TabPanel>
        </Paper>
      )}
    </Box>
  );
};

export default ConversationAnalysis; 