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
  Card,
  CardContent,
  Grid,
  TextField,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
} from '@mui/material';
import { Description, Link, AutoAwesome, ExpandMore, Speed, TrendingUp } from '@mui/icons-material';
import { DocumentUpload } from './DocumentUpload';

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

interface DocumentSummaryResult {
  original_length: number;
  summary_short: string;
  summary_medium: string;
  summary_long: string;
  key_points: string[];
  document_type: string;
  language: string;
  confidence_score: number;
  compression_ratio: number;
  processing_time: number;
  model_info: {
    primary_model: string;
    extractive_method: string;
    pipeline_type: string;
  };
}

const DocumentSummarization: React.FC = () => {
  const [documentFile, setDocumentFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<DocumentSummaryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(0);
  const [mode, setMode] = useState<'file' | 'url'>('file');

  const handleFileSelect = (file: File) => {
    setDocumentFile(file);
    setResult(null);
    setError(null);
    setMode('file');
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const processDocument = async () => {
    if (!documentFile && !url) return;

    setLoading(true);
    setProgress(0);
    setError(null);

    try {
      let jobId: string;

      if (mode === 'file' && documentFile) {
        // Upload document file
        const formData = new FormData();
        formData.append('file', documentFile);

        const uploadResponse = await fetch('http://localhost:8000/upload-document', {
          method: 'POST',
          body: formData,
        });

        if (!uploadResponse.ok) {
          throw new Error('Failed to upload document');
        }

        const uploadData = await uploadResponse.json();
        jobId = uploadData.job_id;
      } else {
        // Process URL
        const urlResponse = await fetch('http://localhost:8000/summarize-url', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: `url=${encodeURIComponent(url)}`,
        });

        if (!urlResponse.ok) {
          throw new Error('Failed to process URL');
        }

        const urlData = await urlResponse.json();
        jobId = urlData.job_id;
      }

      // Poll for results
      while (true) {
        const statusResponse = await fetch(`http://localhost:8000/status/${jobId}`);
        const statusData = await statusResponse.json();

        setProgress(statusData.progress);

        if (statusData.status === 'completed') {
          const resultResponse = await fetch(`http://localhost:8000/result/${jobId}`);
          const summaryResult = await resultResponse.json();
          
          setResult(summaryResult);
          setActiveTab(0);
          break;
        } else if (statusData.status === 'failed') {
          throw new Error(statusData.message);
        }

        await new Promise(resolve => setTimeout(resolve, 2000));
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during document processing');
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  const getCompressionColor = (ratio: number) => {
    if (ratio <= 0.2) return 'success';
    if (ratio <= 0.5) return 'warning';
    return 'error';
  };

  const formatLength = (length: number) => {
    if (length < 1000) return `${length} chars`;
    if (length < 1000000) return `${(length / 1000).toFixed(1)}K chars`;
    return `${(length / 1000000).toFixed(1)}M chars`;
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <Typography variant="h5" fontWeight="bold">
          Document Summarization
        </Typography>
        <Chip
          label="PEGASUS + BART Hybrid"
          color="secondary"
          size="small"
        />
      </Box>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload documents (PDF, DOC, TXT) or provide URLs to get intelligent multi-level summaries using hybrid extractive-abstractive pipeline
      </Typography>

      {/* Input Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ mb: 3 }}>
          <Button
            variant={mode === 'file' ? 'contained' : 'outlined'}
            onClick={() => setMode('file')}
            sx={{ mr: 2 }}
            startIcon={<Description />}
          >
            Upload Document
          </Button>
          <Button
            variant={mode === 'url' ? 'contained' : 'outlined'}
            onClick={() => setMode('url')}
            startIcon={<Link />}
          >
            Summarize URL
          </Button>
        </Box>

        {mode === 'file' ? (
          <DocumentUpload onFileSelect={handleFileSelect} />
        ) : (
          <Box>
            <TextField
              fullWidth
              label="Enter URL to summarize"
              placeholder="https://example.com/article"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              variant="outlined"
              sx={{ mb: 2 }}
            />
            {url && (
              <Alert severity="info" sx={{ mb: 2 }}>
                <Typography variant="body2">
                  URL: {url}
                </Typography>
              </Alert>
            )}
          </Box>
        )}
        
        {((mode === 'file' && documentFile) || (mode === 'url' && url)) && (
          <Box sx={{ mt: 2 }}>
            <Button
              variant="contained"
              onClick={processDocument}
              disabled={loading}
              startIcon={<AutoAwesome />}
              size="large"
            >
              {loading ? 'Processing...' : 'Generate Summary'}
            </Button>
          </Box>
        )}

        {loading && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" sx={{ mb: 1 }}>
              Processing document... {progress}%
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
                label="Summaries"
                icon={<Description />}
                iconPosition="start"
              />
              <Tab
                label="Key Points"
                icon={<TrendingUp />}
                iconPosition="start"
              />
              <Tab
                label="Analysis"
                icon={<Speed />}
                iconPosition="start"
              />
            </Tabs>
          </Box>

          <TabPanel value={activeTab} index={0}>
            <Typography variant="h6" gutterBottom>
              Multi-Level Document Summaries
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card elevation={2} sx={{ bgcolor: 'primary.50' }}>
                  <CardContent>
                    <Typography variant="h6" color="primary" gutterBottom>
                      📋 Quick Summary
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                      {result.summary_short}
                    </Typography>
                    <Chip 
                      label={`${formatLength(result.summary_short.length)}`}
                      size="small"
                      color="primary"
                    />
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      📖 Medium Summary
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                      {result.summary_medium}
                    </Typography>
                    <Chip 
                      label={`${formatLength(result.summary_medium.length)}`}
                      size="small"
                    />
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      📚 Detailed Summary
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                      {result.summary_long}
                    </Typography>
                    <Chip 
                      label={`${formatLength(result.summary_long.length)}`}
                      size="small"
                    />
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            <Typography variant="h6" gutterBottom>
              Key Points & Highlights
            </Typography>
            
            <Card>
              <CardContent>
                {result.key_points.length > 0 ? (
                  <Box>
                    {result.key_points.map((point, index) => (
                      <Box key={index} sx={{ mb: 2 }}>
                        <Typography variant="body1" sx={{ display: 'flex', alignItems: 'flex-start' }}>
                          <Chip 
                            label={index + 1} 
                            size="small" 
                            color="primary" 
                            sx={{ mr: 2, mt: 0.5, minWidth: '32px' }}
                          />
                          {point}
                        </Typography>
                        {index < result.key_points.length - 1 && <Divider sx={{ mt: 2 }} />}
                      </Box>
                    ))}
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No key points extracted
                  </Typography>
                )}
              </CardContent>
            </Card>
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            <Typography variant="h6" gutterBottom>
              Document Analysis & Processing Details
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="h6">Document Statistics</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Original Length:</strong> {formatLength(result.original_length)}
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Document Type:</strong> {result.document_type}
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Language:</strong> {result.language.toUpperCase()}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Processing Time:</strong> {result.processing_time.toFixed(2)}s
                      </Typography>
                    </Box>
                  </AccordionDetails>
                </Accordion>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="h6">Model Information</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Primary Model:</strong> {result.model_info.primary_model}
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Extractive Method:</strong> {result.model_info.extractive_method}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Pipeline:</strong> {result.model_info.pipeline_type}
                      </Typography>
                    </Box>
                  </AccordionDetails>
                </Accordion>
              </Grid>
              
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Quality Metrics
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Chip 
                            label={`Confidence: ${(result.confidence_score * 100).toFixed(1)}%`}
                            color={result.confidence_score >= 0.8 ? 'success' : 'warning'}
                          />
                        </Box>
                      </Grid>
                      <Grid item xs={6}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Chip 
                            label={`Compression: ${(result.compression_ratio * 100).toFixed(1)}%`}
                            color={getCompressionColor(result.compression_ratio)}
                          />
                        </Box>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>
        </Paper>
      )}
    </Box>
  );
};

export default DocumentSummarization; 