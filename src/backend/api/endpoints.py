from flask import Blueprint, request, jsonify
from datetime import datetime
import openai
from .models import (
    ModalityType, TextInput, AudioFeatures, VideoFeatures,
    AnalysisResult, MultimodalPrediction
)
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Create blueprint
api = Blueprint('api', __name__)

def get_current_timestamp():
    return datetime.now().isoformat()

@api.route('/analyze/text', methods=['POST'])
def analyze_text():
    """Analyze text input using both ChatGPT and DeepSeek for comprehensive analysis"""
    data = request.json
    text_input = TextInput(**data)
    
    # ChatGPT Analysis
    try:
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """
                You are an AI trained to analyze text for indicators of autism spectrum disorder.
                Focus on identifying:
                1. Communication patterns
                2. Repetitive language
                3. Social interaction cues
                4. Expression of emotions
                Provide a structured analysis with confidence scores.
                """},
                {"role": "user", "content": text_input.content}
            ]
        )
        gpt_analysis = gpt_response.choices[0].message.content
    except Exception as e:
        gpt_analysis = None
    
    # DeepSeek Analysis
    try:
        deepseek_headers = {
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }
        deepseek_payload = {
            "model": "deepseek-v3",
            "messages": [
                {"role": "system", "content": """
                Analyze the following text for autism spectrum indicators.
                Focus on:
                1. Language complexity
                2. Social understanding
                3. Literal vs. figurative understanding
                4. Topic maintenance
                Provide a structured analysis with confidence scores.
                """},
                {"role": "user", "content": text_input.content}
            ]
        }
        # Note: Implementation would need actual DeepSeek API endpoint
        deepseek_analysis = "DeepSeek analysis placeholder"
    except Exception as e:
        deepseek_analysis = None
    
    # Combine and analyze results
    combined_result = AnalysisResult(
        modality=ModalityType.TEXT,
        confidence_score=0.85,  # This should be calculated based on both analyses
        risk_level="Medium",
        features={
            "gpt_analysis": gpt_analysis,
            "deepseek_analysis": deepseek_analysis,
            "language_patterns": {
                "repetitive_phrases": 0.3,
                "social_context_understanding": 0.7,
                "emotional_expression": 0.6
            }
        },
        timestamp=get_current_timestamp()
    )
    
    return jsonify(combined_result.dict())

@api.route('/analyze/multimodal', methods=['POST'])
def analyze_multimodal():
    """Analyze multiple inputs (text, audio, video, fMRI) for comprehensive assessment"""
    try:
        # Handle file uploads and text input
        text_data = request.form.get('text')
        audio_file = request.files.get('audio')
        video_file = request.files.get('video')
        fmri_file = request.files.get('fmri')
        
        results = []
        
        # Text Analysis
        if text_data:
            text_result = analyze_text()
            results.append(text_result)
        
        # Audio Analysis (placeholder)
        if audio_file:
            audio_result = AnalysisResult(
                modality=ModalityType.AUDIO,
                confidence_score=0.8,
                risk_level="Medium",
                features=AudioFeatures(
                    prosody_score=0.7,
                    rhythm_score=0.6,
                    pitch_variation=0.5,
                    speech_rate=0.8
                ).dict(),
                timestamp=get_current_timestamp()
            )
            results.append(audio_result)
        
        # Video Analysis (placeholder)
        if video_file:
            video_result = AnalysisResult(
                modality=ModalityType.VIDEO,
                confidence_score=0.75,
                risk_level="Medium",
                features=VideoFeatures(
                    eye_contact_score=0.6,
                    facial_expression_score=0.7,
                    emotion_variation=0.5,
                    gaze_patterns=[0.4, 0.6, 0.5]
                ).dict(),
                timestamp=get_current_timestamp()
            )
            results.append(video_result)
        
        # Combine all results
        final_prediction = MultimodalPrediction(
            overall_risk_score=0.65,  # This should be calculated based on all results
            confidence_level=0.8,
            modality_results=results,
            recommendation="Based on the multimodal analysis, further clinical evaluation is recommended.",
            timestamp=get_current_timestamp()
        )
        
        return jsonify(final_prediction.dict())
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": get_current_timestamp()
    })
