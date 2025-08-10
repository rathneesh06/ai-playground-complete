// TranscriptService - Real implementation using the audio backend API
interface ProcessingResult {
  transcript: string;
  diarization: Array<{
    speaker: string;
    text: string;
    start_time: number;
    end_time: number;
    confidence?: number;
  }>;
  summary: string;
  duration: number;
  num_speakers: number;
  language?: string;
  processing_time: number;
  model_info: Record<string, string>;
}

interface ProcessingStatus {
  job_id: string;
  status: string;
  progress: number;
  message: string;
  result?: ProcessingResult;
}

export class TranscriptService {
  private static readonly API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';
  
  /**
   * Transcribe audio file using the real backend API
   */
  static async transcribeAudio(audioFile: File): Promise<string> {
    try {
      // Upload file and start processing
      const jobId = await this.uploadFile(audioFile);
      
      // Poll for completion
      const result = await this.pollForResults(jobId);
      
      return result.transcript;
      
    } catch (error) {
      console.error('Transcription error:', error);
      throw new Error('Failed to transcribe audio. Please try again.');
    }
  }

  /**
   * Get complete conversation analysis including diarization
   */
  static async analyzeConversation(audioFile: File): Promise<{
    transcript: string;
    diarization: Array<{
      speaker: string;
      text: string;
      timestamp: string;
    }>;
    summary: string;
  }> {
    try {
      // Upload file and start processing
      const jobId = await this.uploadFile(audioFile);
      
      // Poll for completion
      const result = await this.pollForResults(jobId);
      
      // Convert backend format to frontend format
      const diarization = result.diarization.map(segment => ({
        speaker: segment.speaker,
        text: segment.text,
        timestamp: this.formatTimestamp(segment.start_time)
      }));
      
      return {
        transcript: result.transcript,
        diarization,
        summary: result.summary
      };
      
    } catch (error) {
      console.error('Analysis error:', error);
      throw new Error('Failed to analyze conversation. Please try again.');
    }
  }

  /**
   * Upload file to backend and get job ID
   */
  private static async uploadFile(audioFile: File): Promise<string> {
    const formData = new FormData();
    formData.append('file', audioFile);

    const response = await fetch(`${this.API_BASE}/upload-audio`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.job_id;
  }

  /**
   * Poll for processing results with progress updates
   */
  private static async pollForResults(
    jobId: string,
    onProgress?: (progress: number, message: string) => void
  ): Promise<ProcessingResult> {
    const maxAttempts = 120; // 10 minutes max (5 second intervals)
    let attempts = 0;

    while (attempts < maxAttempts) {
      try {
        // Check status
        const statusResponse = await fetch(`${this.API_BASE}/status/${jobId}`);
        
        if (!statusResponse.ok) {
          throw new Error(`Status check failed: ${statusResponse.status}`);
        }

        const status: ProcessingStatus = await statusResponse.json();

        // Update progress if callback provided
        if (onProgress) {
          onProgress(status.progress, status.message);
        }

        // Check if completed
        if (status.status === 'completed') {
          if (status.result) {
            return status.result;
          } else {
            // Get result from result endpoint
            const resultResponse = await fetch(`${this.API_BASE}/result/${jobId}`);
            
            if (!resultResponse.ok) {
              throw new Error(`Result fetch failed: ${resultResponse.status}`);
            }
            
            return await resultResponse.json();
          }
        }

        // Check if failed
        if (status.status === 'failed') {
          throw new Error(`Processing failed: ${status.message}`);
        }

        // Wait before next poll
        await this.delay(5000); // 5 seconds
        attempts++;

      } catch (error) {
        console.error('Polling error:', error);
        
        // Retry a few times for network errors
        if (attempts < 3) {
          await this.delay(2000);
          attempts++;
          continue;
        }
        
        throw error;
      }
    }

    throw new Error('Processing timeout. Please try again with a shorter audio file.');
  }

  /**
   * Generate a summary of the transcript (now handled by backend)
   */
  static async generateSummary(transcript: string): Promise<string> {
    // This is now handled in the main analysis pipeline
    // But we can provide a fallback for standalone summarization
    try {
      // For now, return a simple summary
      // In future, could add a separate summarization endpoint
      const sentences = transcript.split(/[.!?]+/).filter(s => s.trim().length > 0);
      
      if (sentences.length <= 3) {
        return transcript;
      }
      
      // Simple extractive summary - take first and last sentences plus one from middle
      const summary = [
        sentences[0],
        sentences[Math.floor(sentences.length / 2)],
        sentences[sentences.length - 1]
      ].join('. ') + '.';
      
      return summary;
      
    } catch (error) {
      console.error('Summary generation error:', error);
      return 'Summary could not be generated.';
    }
  }

  /**
   * Health check for the backend service
   */
  static async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.API_BASE}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      return response.ok;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  /**
   * Format timestamp for display
   */
  private static formatTimestamp(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  /**
   * Utility function for delays
   */
  private static delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Check if backend is available and fallback to mock if needed
   */
  static async isBackendAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.API_BASE}/`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      return response.ok;
    } catch (error) {
      console.warn('Backend not available, using mock data');
      return false;
    }
  }

  /**
   * Fallback mock implementation for development/demo
   */
  static async mockTranscribeAudio(): Promise<string> {
    await this.delay(2000);
    
    return `Speaker 1: Hello, thank you for calling our customer service. How can I help you today?

Speaker 2: Hi, I'm having an issue with my recent order. I placed an order last week but haven't received any shipping confirmation yet.

Speaker 1: I'm sorry to hear about that. Let me look into your order for you. Can you please provide me with your order number?

Speaker 2: Sure, it's order number 12345. I placed it on Monday and paid extra for expedited shipping.

Speaker 1: Thank you for that information. Let me check our system... I can see your order here. It looks like there was a delay in processing due to inventory issues, but I can confirm that your order shipped yesterday. You should receive a tracking email within the next hour.

Speaker 2: Oh, that's a relief! Thank you for checking. Can you tell me when I should expect to receive it?

Speaker 1: Based on the expedited shipping you selected, you should receive your order by tomorrow afternoon. I'll also make sure to apply a 10% discount to your next order as an apology for the confusion.

Speaker 2: That's very generous, thank you so much for your help! I really appreciate the excellent customer service.

Speaker 1: You're very welcome! Is there anything else I can help you with today?

Speaker 2: No, that covers everything. Thank you again and have a great day!

Speaker 1: Thank you, you too! Take care!`;
  }
} 