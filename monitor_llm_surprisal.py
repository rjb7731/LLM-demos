import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import gradio as gr
import time
import math

class StorySurpriseScorer:
    """
    Scores text based on how surprising it is to a language model.
    Lower perplexity = more predictable = lower score
    Higher perplexity = more surprising = higher score
    """
    
    def __init__(self, model_name="google/gemma-3-1b-it"): # <-- CHANGED MODEL
        """
        Initialize with a HuggingFace model.
        """
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # --- NEW: Define the prompt that gives the model context ---
        # This MUST match the context you give the user in the UI.
        self.context_prompt = "Write a unique short story about a bear:\n\n"
    
    def calculate_perplexity(self, text):
        """Calculate perplexity of the text."""
        if not text.strip():
            return 0.0
            
        # --- NEW: Combine prompt and user text ---
        full_text = self.context_prompt + text
        encodings = self.tokenizer(full_text, return_tensors="pt")
        input_ids = encodings.input_ids
        
        # --- NEW: Calculate length of the prompt in tokens ---
        # We need this to ignore the prompt when calculating loss.
        prompt_encodings = self.tokenizer(self.context_prompt, return_tensors="pt")
        prompt_length = prompt_encodings.input_ids.shape[1]

        # --- NEW: Create labels that ignore the prompt ---
        # We set the prompt tokens to -100, so they are ignored by the loss function.
        # This is the standard way to calculate perplexity on a "continuation".
        labels = input_ids.clone()
        labels[:, :prompt_length] = -100
        
        with torch.no_grad():
            # --- MODIFIED: Pass the new labels ---
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
        
        perplexity = torch.exp(loss).item()
        return perplexity
    
    def get_token_surprises(self, text):
        """Get surprise score for each token."""
        if not text.strip():
            return []
            
        # --- NEW: Combine prompt and user text ---
        full_text = self.context_prompt + text
        encodings = self.tokenizer(full_text, return_tensors="pt")
        input_ids = encodings.input_ids

        # --- NEW: Calculate length of the prompt in tokens ---
        prompt_encodings = self.tokenizer(self.context_prompt, return_tensors="pt")
        prompt_length = prompt_encodings.input_ids.shape[1]
        
        token_surprises = []
        
        with torch.no_grad():
            # --- MODIFIED: Start the loop AFTER the prompt ---
            # We start at prompt_length to only score the user's words.
            for i in range(prompt_length, input_ids.shape[1]):
                context = input_ids[:, :i]
                target = input_ids[:, i]
                
                outputs = self.model(context)
                logits = outputs.logits[0, -1, :]
                
                probs = torch.softmax(logits, dim=-1)
                actual_prob = probs[target].item()
                
                surprise = -np.log(actual_prob + 1e-10)
                
                token = self.tokenizer.decode([target.item()])
                token_surprises.append((token, surprise, actual_prob))
        
        return token_surprises
    
    def _interpret_score(self, score):
        """Helper function to give qualitative feedback."""
        if score < 20:
            return "Very predictable. The AI saw this coming a mile away!"
        elif score < 40:
            return "A bit cliché, but has a spark!"
        elif score < 60:
            return "Moderately creative! You're breaking the mold."
        elif score < 80:
            return "Quite surprising! You've got the AI on its toes."
        else:
            return "Extremely surprising! A true work of human originality!"
    
    def score_story(self, text):
        """Score a story based on its surprise factor."""
        if not text.strip():
            return {
                'creativity_score': 0,
                'perplexity': 0,
                'interpretation': 'Start writing!',
                'token_analysis': [],
                'most_surprising': [] 
            }
        
        perplexity = self.calculate_perplexity(text)
        
        # This log-based formula is good. Keep it.
        creativity_score = min(100, max(0, 20 * (math.log(max(1, perplexity)) - math.log(10))))
        
        token_surprises = self.get_token_surprises(text)
        
        return {
            'creativity_score': round(creativity_score, 2),
            'perplexity': round(perplexity, 2),
            'interpretation': self._interpret_score(creativity_score),
            'token_analysis': token_surprises,
            'most_surprising': sorted(token_surprises, key=lambda x: x[1], reverse=True)[:5]
        }


# Initialize scorer globally
print("Initializing model...")
# --- MODIFIED: Change the model name here too! ---
scorer = StorySurpriseScorer(model_name="google/gemma-3-1b-it") 
print("Model loaded!")

def analyze_text(text):
    """Analyze text and return formatted results."""
    result = scorer.score_story(text)
    
    # Format creativity score with color
    score = result['creativity_score']
    if score < 20:
        color = "#ff4444"
    elif score < 40:
        color = "#ff8844"
    elif score < 60:
        color = "#ffcc44"
    elif score < 80:
        color = "#88ff44"
    else:
        color = "#44ff88"
    
    score_html = f"""
    <div style="text-align: center; padding: 20px;">
        <h2>Creativity Score</h2>
        <div style="font-size: 72px; font-weight: bold; color: {color};">
            {score}/100
        </div>
        <p style="font-size: 24px; margin-top: 10px;">
            {result['interpretation']}
        </p>
        <p style="font-size: 16px; color: #666; margin-top: 10px;">
            Perplexity: {result['perplexity']}
        </p>
    </div>
    """
    
    # Format most surprising tokens
    if result['most_surprising']:
        tokens_html = "<h3>🎯 Most Surprising Tokens:</h3><ul>"
        for token, surprise, prob in result['most_surprising']:
            tokens_html += f"<li><code>{token}</code> - Surprise: {surprise:.2f} (Probability: {prob:.4f})</li>"
        tokens_html += "</ul>"
    else:
        tokens_html = ""
    
    word_count = len(text.split())
    stats_html = f"<p><strong>Word count:</strong> {word_count}</p>"
    
    return score_html, stats_html + tokens_html

# Create Gradio interface
with gr.Blocks(title="Beat the LLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎮 Beat the LLM - Creative Writing Challenge
    
    ### Can you write something so creative that the AI (Gemma 1B) doesn't see it coming?
    
    **Your mission:** Write a story about a bear that surprises the language model.
    
    The AI has been trained on trillions of words from the internet. It "expects" common phrases and clichés.
                
    Write something better than the average of the internet!
                
    **Your goal is to write something genuinely original and unexpected!**
    
    💡 **Tips:**
    - Avoid clichés like "Once upon a time"
    - Use unusual word combinations
    - Create unexpected metaphors and imagery
    - Break narrative conventions
    - Be weird, experimental, and bold!
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Write your story about a bear:",
                placeholder="The bear...",
                lines=15,
                max_lines=20
            )
            
            analyze_btn = gr.Button("🎯 Analyze My Creativity!", variant="primary", size="lg")
            
            gr.Markdown("""
            <div style="margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 8px;">
            <strong>How it works:</strong><br>
            The system measures how "surprised" Gemma-1b is by your word choices. 
            Higher surprise = more creative and original writing!
            </div>
            """)
        
        with gr.Column(scale=1):
            score_output = gr.HTML(label="Score")
            details_output = gr.HTML(label="Details")
    
    # Auto-update on button click
    analyze_btn.click(
        fn=analyze_text,
        inputs=[text_input],
        outputs=[score_output, details_output]
    )
    
    # Also update as user types (with slight delay)
    text_input.change(
        fn=analyze_text,
        inputs=[text_input],
        outputs=[score_output, details_output]
    )
    
    gr.Markdown("""
    ---
    ### 📊 What do the scores mean?
    
    - **0-20**: Very predictable - the LLM has seen this before
    - **20-40**: Somewhat predictable - you can do better!
    - **40-60**: Moderately creative - decent originality
    - **60-80**: Quite surprising - great creativity!
    - **80-100**: EXTREMELY SURPRISING - You've created something truly unique!
    
    ### 🎨 Example prompts to try:
    - Write a story about a bear (original prompt)
    - Describe a dragon meeting a programmer
    - Tell a story about time traveling backwards
    - Write about an emotion that doesn't exist yet
    """)

if __name__ == "__main__":
    demo.launch(share=True)