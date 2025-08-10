import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Container,
  Paper,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
} from '@mui/material';
import { SmartToy, Logout } from '@mui/icons-material';
import { useAuth } from '../contexts/AuthContext';
import ConversationAnalysis from './skills/ConversationAnalysis';
import ImageAnalysis from './skills/ImageAnalysis';
import DocumentSummarization from './skills/DocumentSummarization';

export type SkillType = 'conversation' | 'image' | 'summarization';

const skills = [
  { value: 'conversation', label: 'Conversation Analysis', description: 'Upload audio, transcribe, and diarize speech' },
  { value: 'image', label: 'Image Analysis', description: 'Upload images and generate descriptions' },
  { value: 'summarization', label: 'Document/URL Summarization', description: 'Summarize PDFs, documents, or web pages' },
];

const Dashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const [selectedSkill, setSelectedSkill] = useState<SkillType>('conversation');

  const handleSkillChange = (event: SelectChangeEvent<SkillType>) => {
    setSelectedSkill(event.target.value as SkillType);
  };

  const renderSkillComponent = () => {
    switch (selectedSkill) {
      case 'conversation':
        return <ConversationAnalysis />;
      case 'image':
        return <ImageAnalysis />;
      case 'summarization':
        return <DocumentSummarization />;
      default:
        return null;
    }
  };

  return (
    <>
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <SmartToy sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            AI Playground
          </Typography>
          <Typography variant="body2" sx={{ mr: 2 }}>
            Welcome, {user?.name}
          </Typography>
          <Button
            color="inherit"
            onClick={logout}
            startIcon={<Logout />}
          >
            Logout
          </Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h4" gutterBottom fontWeight="bold">
            AI Skills Playground
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            Select an AI skill to explore its capabilities
          </Typography>

          <FormControl fullWidth sx={{ mb: 3 }}>
            <InputLabel id="skill-select-label">Select AI Skill</InputLabel>
            <Select
              labelId="skill-select-label"
              id="skill-select"
              value={selectedSkill}
              label="Select AI Skill"
              onChange={handleSkillChange}
            >
              {skills.map((skill) => (
                <MenuItem key={skill.value} value={skill.value}>
                  <Box>
                    <Typography variant="body1" fontWeight="medium">
                      {skill.label}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {skill.description}
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Paper>

        <Paper sx={{ minHeight: 500 }}>
          {renderSkillComponent()}
        </Paper>
      </Container>
    </>
  );
};

export default Dashboard; 