from flask import Flask, render_template, request, jsonify, stream_with_context, Response
from groq import Groq
import json
import os
from datetime import datetime

app = Flask(__name__)

# Initialize Groq clients with two API keys
PRIMARY_API_KEY = 'test'
ANALYZER_API_KEY = 'test'

primary_client = Groq(api_key=PRIMARY_API_KEY)
analyzer_client = Groq(api_key=ANALYZER_API_KEY)

MODEL_NAME = "llama-3.3-70b-versatile"  # Using Llama 3.3 70B model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_prompt():
    try:
        data = request.json
        user_prompt = data.get('prompt', '')
        
        if not user_prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        # Step 1: Send to Primary LLM
        primary_response = get_primary_llm_response(user_prompt)
        
        # Step 2: Send to Analyzer LLM
        analyzer_response = get_analyzer_llm_response(user_prompt, primary_response)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'user_prompt': user_prompt,
            'primary_llm': primary_response,
            'analyzer_llm': analyzer_response
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_primary_llm_response(user_prompt):
    """Get response from Primary LLM"""
    try:
        # System prompt for primary LLM
        system_prompt = """You are a helpful AI assistant. Provide detailed, accurate, and well-reasoned responses.
        Show your thinking process clearly and explain your reasoning step by step."""
        
        # Call Groq API for primary response
        chat_completion = primary_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=MODEL_NAME,
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            stream=False
        )
        
        response_content = chat_completion.choices[0].message.content
        
        # Extract thinking/reasoning (simulate analysis)
        thinking = analyze_response_thinking(response_content)
        
        return {
            'response': response_content,
            'thinking': thinking,
            'model': MODEL_NAME,
            'tokens_used': chat_completion.usage.total_tokens,
            'prompt_tokens': chat_completion.usage.prompt_tokens,
            'completion_tokens': chat_completion.usage.completion_tokens,
            'finish_reason': chat_completion.choices[0].finish_reason
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'response': None,
            'thinking': f'Error occurred: {str(e)}'
        }

def get_analyzer_llm_response(user_prompt, primary_response):
    """Get analysis and suggestions from Analyzer LLM"""
    try:
        # System prompt for analyzer
        analysis_system_prompt = """You are an expert prompt engineer and AI response analyzer. 
        Your job is to:
        1. Analyze the user's original prompt for clarity, specificity, and effectiveness
        2. Evaluate the primary LLM's response for accuracy, completeness, and relevance
        3. Identify any gaps, ambiguities, or areas for improvement
        4. Suggest an improved version of the prompt that would yield better results
        5. Explain your reasoning clearly
        
        IMPORTANT: You MUST provide your analysis in the following structure with clear headers:
        
        **Prompt Analysis:**
        [Evaluate the original prompt]
        
        **Response Quality:**
        [Assess the primary LLM's response]
        
        **Identified Issues:**
        - [List any problems or limitations]
        
        **Suggested Improvements:**
        - [Provide specific recommendations]
        
        **Refined Prompt:**
        [Provide a complete, improved version of the prompt that incorporates all your suggestions. This should be a ready-to-use prompt that the user can copy and try immediately.]
        
        Make sure the Refined Prompt section contains a complete, actionable prompt."""
        
        # Construct analysis prompt
        analysis_prompt = f"""Original User Prompt:
{user_prompt}

Primary LLM Response:
{primary_response.get('response', 'No response available')}

Primary LLM Thinking Process:
{primary_response.get('thinking', 'No thinking data available')}

Please analyze this interaction and provide comprehensive feedback and suggestions."""
        
        # Call Groq API for analysis
        chat_completion = analyzer_client.chat.completions.create(
            messages=[
                {"role": "system", "content": analysis_system_prompt},
                {"role": "user", "content": analysis_prompt}
            ],
            model=MODEL_NAME,
            temperature=0.8,
            max_tokens=2048,
            top_p=1,
            stream=False
        )
        
        analysis_content = chat_completion.choices[0].message.content
        
        # Parse analysis into structured format
        parsed_analysis = parse_analysis(analysis_content)
        
        return {
            'analysis': analysis_content,
            'parsed': parsed_analysis,
            'model': MODEL_NAME,
            'tokens_used': chat_completion.usage.total_tokens,
            'prompt_tokens': chat_completion.usage.prompt_tokens,
            'completion_tokens': chat_completion.usage.completion_tokens,
            'reasoning': extract_reasoning(analysis_content)
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'analysis': None,
            'reasoning': f'Error occurred during analysis: {str(e)}'
        }

def analyze_response_thinking(response_content):
    """Analyze the thinking process from response"""
    thinking = []
    
    # Check for structured thinking
    if "step" in response_content.lower() or "first" in response_content.lower():
        thinking.append("Response follows a structured, step-by-step approach")
    
    # Check for reasoning indicators
    reasoning_words = ["because", "therefore", "thus", "since", "as a result", "consequently"]
    if any(word in response_content.lower() for word in reasoning_words):
        thinking.append("Response includes causal reasoning and explanations")
    
    # Check for examples
    if "example" in response_content.lower() or "for instance" in response_content.lower():
        thinking.append("Response provides concrete examples")
    
    # Check response length and detail
    word_count = len(response_content.split())
    if word_count > 200:
        thinking.append(f"Comprehensive response with {word_count} words")
    elif word_count > 100:
        thinking.append(f"Moderate-length response with {word_count} words")
    else:
        thinking.append(f"Concise response with {word_count} words")
    
    return thinking

def parse_analysis(analysis_content):
    """Parse the analysis into structured components"""
    parsed = {
        'prompt_analysis': '',
        'response_quality': '',
        'identified_issues': [],
        'suggested_improvements': [],
        'refined_prompt': ''
    }
    
    sections = {
        'prompt analysis': 'prompt_analysis',
        'response quality': 'response_quality',
        'identified issues': 'identified_issues',
        'suggested improvements': 'suggested_improvements',
        'refined prompt': 'refined_prompt'
    }
    
    current_section = None
    lines = analysis_content.split('\n')
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if line is a section header
        for section_name, section_key in sections.items():
            if section_name in line_lower and ':' in line:
                current_section = section_key
                # Get content after the colon if any
                content_after = line.split(':', 1)[1].strip() if ':' in line else ''
                if content_after:
                    if isinstance(parsed[current_section], list):
                        parsed[current_section].append(content_after)
                    else:
                        parsed[current_section] = content_after
                break
        else:
            # Add content to current section
            if current_section and line.strip():
                if isinstance(parsed[current_section], list):
                    if line.strip().startswith('-') or line.strip().startswith('•'):
                        parsed[current_section].append(line.strip()[1:].strip())
                    elif line.strip()[0].isdigit() and '.' in line[:3]:
                        parsed[current_section].append(line.strip().split('.', 1)[1].strip())
                else:
                    parsed[current_section] += ' ' + line.strip()
    
    return parsed

def extract_reasoning(analysis_content):
    """Extract key reasoning points from analysis"""
    reasoning = []
    
    # Look for reasoning indicators
    lines = analysis_content.split('\n')
    for line in lines:
        if any(word in line.lower() for word in ['because', 'therefore', 'this suggests', 'indicates', 'shows that']):
            reasoning.append(line.strip())
    
    # If no explicit reasoning found, extract key points
    if not reasoning:
        for line in lines:
            if line.strip() and len(line.split()) > 10:
                reasoning.append(line.strip())
        reasoning = reasoning[:5]  # Limit to 5 key points
    
    return reasoning

if __name__ == '__main__':
    app.run(debug=True, port=5000)