// DiarizationService - Custom speaker diarization for up to 2 speakers
interface DiarizedSegment {
  speaker: string;
  text: string;
  timestamp: string;
}

export class DiarizationService {
  /**
   * Custom speaker diarization logic for up to 2 speakers
   * This is a simplified implementation that uses pattern matching
   * In production, you might use more sophisticated ML models
   */
  static async diarizeTranscript(transcript: string): Promise<DiarizedSegment[]> {
    try {
      await this.delay(1500); // Simulate processing time
      
      // Split transcript into segments based on speaker patterns
      const segments = this.parseTranscriptSegments(transcript);
      
      // Apply speaker identification logic
      const diarizedSegments = this.identifySpeakers(segments);
      
      return diarizedSegments;
      
    } catch (error) {
      console.error('Diarization error:', error);
      throw new Error('Failed to perform speaker diarization. Please try again.');
    }
  }

  /**
   * Parse transcript into individual speech segments
   */
  private static parseTranscriptSegments(transcript: string): Array<{text: string, index: number}> {
    // Split by paragraphs and filter out empty lines
    const lines = transcript.split('\n\n').filter(line => line.trim().length > 0);
    
    return lines.map((line, index) => ({
      text: line.trim(),
      index
    }));
  }

  /**
   * Identify speakers using various heuristics
   */
  private static identifySpeakers(segments: Array<{text: string, index: number}>): DiarizedSegment[] {
    const result: DiarizedSegment[] = [];
    
    for (const segment of segments) {
      const speakerInfo = this.determineSpeaker(segment.text, segment.index);
      
      result.push({
        speaker: speakerInfo.speaker,
        text: speakerInfo.cleanedText,
        timestamp: this.generateTimestamp(segment.index),
      });
    }
    
    return result;
  }

  /**
   * Determine speaker based on content analysis and patterns
   */
  private static determineSpeaker(text: string, index: number): {speaker: string, cleanedText: string} {
    // Check if text already has speaker labels
    const speakerMatch = text.match(/^(Speaker \d+):\s*(.+)/);
    if (speakerMatch) {
      return {
        speaker: speakerMatch[1],
        cleanedText: speakerMatch[2]
      };
    }

    // Apply heuristics to identify speakers
    let speaker = 'Speaker 1';
    
    // Business/Customer service patterns
    const customerServicePatterns = [
      /thank you for calling/i,
      /how can I help/i,
      /let me check/i,
      /I can see/i,
      /I'll make sure/i,
      /is there anything else/i
    ];
    
    const customerPatterns = [
      /I'm having an issue/i,
      /I placed an order/i,
      /can you tell me/i,
      /that's very generous/i,
      /thank you so much/i,
      /no, that covers everything/i
    ];

    // Check patterns
    const isCustomerService = customerServicePatterns.some(pattern => pattern.test(text));
    const isCustomer = customerPatterns.some(pattern => pattern.test(text));
    
    if (isCustomerService) {
      speaker = 'Speaker 1';
    } else if (isCustomer) {
      speaker = 'Speaker 2';
    } else {
      // Alternate speakers if no clear pattern
      speaker = index % 2 === 0 ? 'Speaker 1' : 'Speaker 2';
    }

    return {
      speaker,
      cleanedText: text
    };
  }

  /**
   * Generate mock timestamps for segments
   */
  private static generateTimestamp(index: number): string {
    const baseTime = new Date('2024-01-01T10:00:00');
    const segmentDuration = 15; // Average 15 seconds per segment
    
    const timestamp = new Date(baseTime.getTime() + (index * segmentDuration * 1000));
    
    return timestamp.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  }

  /**
   * Utility function to simulate processing delays
   */
  private static delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Advanced speaker identification using voice characteristics simulation
   * This would typically involve audio processing and ML models
   */
  static async advancedDiarization(audioFile: File): Promise<DiarizedSegment[]> {
    // This is where you would implement more sophisticated diarization
    // using libraries like pyannote.audio, resemblyzer, or cloud services
    
    // For now, we'll use the transcript-based approach
    throw new Error('Advanced audio-based diarization not implemented in demo');
  }
} 