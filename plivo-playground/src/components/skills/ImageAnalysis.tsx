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
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import { Image as ImageIcon, Psychology, Visibility, ExpandMore } from '@mui/icons-material';
import { ImageUpload } from './ImageUpload';

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

interface ImageAnalysisResult {
  primary_caption: string;
  detailed_description: string;
  hyper_detailed_caption: string;
  objects_detected: Array<{
    label: string;
    confidence: number;
    bbox?: any;
  }>;
  spatial_relationships: string[];
  context_analysis: {
    scene_type: string;
    complexity: string;
    style: string;
    lighting: string;
  };
  confidence_scores: {
    primary_caption: number;
    object_detection: number;
    overall_analysis: number;
  };
  processing_time: number;
  model_info: {
    primary_model: string;
    object_detector: string;
    enhancement_method: string;
  };
}

const ImageAnalysis: React.FC = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<ImageAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(0);

  const handleFileSelect = (file: File) => {
    setImageFile(file);
    setResult(null);
    setError(null);

    // Create image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const processImage = async () => {
    if (!imageFile) return;

    setLoading(true);
    setProgress(0);
    setError(null);

    try {
      // Upload image to backend
      const formData = new FormData();
      formData.append('file', imageFile);

      const uploadResponse = await fetch('http://localhost:8000/upload-image', {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('Failed to upload image');
      }

      const { job_id } = await uploadResponse.json();

      // Poll for results
      while (true) {
        const statusResponse = await fetch(`http://localhost:8000/status/${job_id}`);
        const statusData = await statusResponse.json();

        setProgress(statusData.progress);

        if (statusData.status === 'completed') {
          const resultResponse = await fetch(`http://localhost:8000/result/${job_id}`);
          const analysisResult = await resultResponse.json();
          
          setResult(analysisResult);
          setActiveTab(0);
          break;
        } else if (statusData.status === 'failed') {
          throw new Error(statusData.message);
        }

        await new Promise(resolve => setTimeout(resolve, 2000));
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during image analysis');
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'success';
    if (confidence >= 0.7) return 'warning';
    return 'error';
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <Typography variant="h5" fontWeight="bold">
          Image Analysis
        </Typography>
        <Chip
          label="InternVL3-Style MLLM"
          color="primary"
          size="small"
        />
      </Box>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Upload an image to get hyper-detailed AI-powered descriptions with object detection and spatial analysis
      </Typography>

      {/* Image Upload Section */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <ImageUpload onFileSelect={handleFileSelect} />
        
        {imageFile && (
          <Box sx={{ mt: 2 }}>
            {imagePreview && (
              <Box sx={{ mb: 2, textAlign: 'center' }}>
                <img 
                  src={imagePreview} 
                  alt="Preview" 
                  style={{ 
                    maxWidth: '300px', 
                    maxHeight: '200px', 
                    objectFit: 'contain',
                    borderRadius: '8px',
                    border: '1px solid #ddd'
                  }} 
                />
              </Box>
            )}
            
            <Alert severity="info" sx={{ mb: 2 }}>
              <Typography variant="body2">
                Selected: {imageFile.name} ({(imageFile.size / 1024 / 1024).toFixed(2)} MB)
              </Typography>
            </Alert>
            
            <Button
              variant="contained"
              onClick={processImage}
              disabled={loading}
              startIcon={<Psychology />}
              size="large"
            >
              {loading ? 'Analyzing...' : 'Analyze Image'}
            </Button>
          </Box>
        )}

        {loading && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" sx={{ mb: 1 }}>
              Processing image... {progress}%
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
                label="Captions"
                icon={<ImageIcon />}
                iconPosition="start"
              />
              <Tab
                label="Objects & Spatial"
                icon={<Visibility />}
                iconPosition="start"
              />
              <Tab
                label="Analysis Details"
                icon={<Psychology />}
                iconPosition="start"
              />
            </Tabs>
          </Box>

          <TabPanel value={activeTab} index={0}>
            <Typography variant="h6" gutterBottom>
              AI-Generated Image Descriptions
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card elevation={2}>
                  <CardContent>
                    <Typography variant="h6" color="primary" gutterBottom>
                      Hyper-Detailed Caption (IIW-Style)
                    </Typography>
                    <Typography variant="body1" sx={{ fontStyle: 'italic', mb: 2 }}>
                      "{result.hyper_detailed_caption}"
                    </Typography>
                    <Chip 
                      label={`Confidence: ${(result.confidence_scores.overall_analysis * 100).toFixed(1)}%`}
                      color={getConfidenceColor(result.confidence_scores.overall_analysis)}
                      size="small"
                    />
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Primary Caption
                    </Typography>
                    <Typography variant="body1">
                      {result.primary_caption}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Detailed Description
                    </Typography>
                    <Typography variant="body1">
                      {result.detailed_description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            <Typography variant="h6" gutterBottom>
              Object Detection & Spatial Analysis
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Detected Objects
                    </Typography>
                    {result.objects_detected.length > 0 ? (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                        {result.objects_detected.map((obj, index) => (
                          <Chip
                            key={index}
                            label={`${obj.label} (${(obj.confidence * 100).toFixed(1)}%)`}
                            color={getConfidenceColor(obj.confidence)}
                            variant="outlined"
                          />
                        ))}
                      </Box>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        No objects detected with high confidence
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Spatial Relationships
                    </Typography>
                    {result.spatial_relationships.length > 0 ? (
                      <Box>
                        {result.spatial_relationships.map((relationship, index) => (
                          <Typography key={index} variant="body2" sx={{ mb: 1 }}>
                            • {relationship}
                          </Typography>
                        ))}
                      </Box>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        No clear spatial relationships detected
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            <Typography variant="h6" gutterBottom>
              Analysis Details & Model Information
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="h6">Context Analysis</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Scene Type:</strong> {result.context_analysis.scene_type}
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Complexity:</strong> {result.context_analysis.complexity}
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Style:</strong> {result.context_analysis.style}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Lighting:</strong> {result.context_analysis.lighting}
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
                        <strong>Object Detector:</strong> {result.model_info.object_detector}
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Enhancement:</strong> {result.model_info.enhancement_method}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Processing Time:</strong> {result.processing_time.toFixed(2)}s
                      </Typography>
                    </Box>
                  </AccordionDetails>
                </Accordion>
              </Grid>
              
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Confidence Scores
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={4}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Chip 
                            label={`Caption: ${(result.confidence_scores.primary_caption * 100).toFixed(1)}%`}
                            color={getConfidenceColor(result.confidence_scores.primary_caption)}
                          />
                        </Box>
                      </Grid>
                      <Grid item xs={4}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Chip 
                            label={`Objects: ${(result.confidence_scores.object_detection * 100).toFixed(1)}%`}
                            color={getConfidenceColor(result.confidence_scores.object_detection)}
                          />
                        </Box>
                      </Grid>
                      <Grid item xs={4}>
                        <Box sx={{ textAlign: 'center' }}>
                          <Chip 
                            label={`Overall: ${(result.confidence_scores.overall_analysis * 100).toFixed(1)}%`}
                            color={getConfidenceColor(result.confidence_scores.overall_analysis)}
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

export default ImageAnalysis; 