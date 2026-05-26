import openai
import json
import pandas as pd
from typing import List, Dict, Union
import time
from dataclasses import dataclass
import os
from pathlib import Path

# Set up OpenAI clien
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")  # Make sure to set your API key
)

@dataclass
class SentimentResult:
    """Data class to store sentiment analysis results"""
    text: str
    sentiment: str
    confidence: float
    reasoning: str = ""
    emotions: Dict[str, float] = None

class SentimentAnalyzer:
    """OpenAI-powered sentiment analysis class"""
    
    def __init__(self, model="gpt-5.4-mini"):
        self.model = model
        self.client = client
    
    def analyze_simple(self, text: str) -> str:
        """Simple sentiment analysis returning just positive/negative/neutral"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a sentiment analysis expert. Analyze the sentiment of the given text and respond with only one word: 'positive', 'negative', or 'neutral'."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze the sentiment of this text: {text}"
                    }
                ],
                temperature=0.1,
                #max_tokens=10
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"Error in simple analysis: {e}")
            return "error"
    
    def analyze_detailed(self, text: str) -> SentimentResult:
        """Detailed sentiment analysis with confidence scores and reasoning"""
        prompt = f"""
        Analyze the sentiment of the following text and provide a detailed analysis.
        
        Text: "{text}"
        
        Please respond with a JSON object containing:
        - sentiment: one of "positive", "negative", "neutral"
        - confidence: a float between 0 and 1 indicating confidence in the sentiment
        - reasoning: a brief explanation of why you classified it this way
        - emotions: an object with emotion scores (joy, anger, fear, sadness, surprise, disgust) each between 0 and 1
        
        Example format:
        {{
            "sentiment": "positive",
            "confidence": 0.85,
            "reasoning": "The text expresses enthusiasm and satisfaction",
            "emotions": {{
                "joy": 0.8,
                "anger": 0.1,
                "fear": 0.0,
                "sadness": 0.0,
                "surprise": 0.2,
                "disgust": 0.0
            }}
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sentiment analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                #max_tokens=300
            )
            
            # Parse the JSON response
            result_json = json.loads(response.choices[0].message.content)
            
            return SentimentResult(
                text=text,
                sentiment=result_json["sentiment"],
                confidence=result_json["confidence"],
                reasoning=result_json["reasoning"],
                emotions=result_json["emotions"]
            )
            
        except Exception as e:
            print(f"Error in detailed analysis: {e}")
            return SentimentResult(
                text=text,
                sentiment="error",
                confidence=0.0,
                reasoning=f"Analysis failed: {str(e)}"
            )
    
    def analyze_batch(self, texts: List[str], delay: float = 0.5) -> List[SentimentResult]:
        """Analyze multiple texts with rate limiting"""
        results = []
        for i, text in enumerate(texts):
            print(f"Analyzing text {i+1}/{len(texts)}")
            result = self.analyze_detailed(text)
            results.append(result)
            
            # Rate limiting to avoid API limits
            if i < len(texts) - 1:
                time.sleep(delay)
        
        return results
    
    def analyze_with_context(self, text: str, context: str = "") -> SentimentResult:
        """Analyze sentiment with additional context"""
        prompt = f"""
        Analyze the sentiment of the following text, considering the provided context.
        
        Context: {context}
        Text to analyze: "{text}"
        
        Provide your analysis as JSON with sentiment, confidence, reasoning, and emotions.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert sentiment analyst. Consider context when analyzing sentiment. Respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result_json = json.loads(response.choices[0].message.content)
            return SentimentResult(
                text=text,
                sentiment=result_json["sentiment"],
                confidence=result_json["confidence"],
                reasoning=result_json["reasoning"],
                emotions=result_json.get("emotions", {})
            )
            
        except Exception as e:
            print(f"Error in contextual analysis: {e}")
            return SentimentResult(text=text, sentiment="error", confidence=0.0)

def save_results_to_csv(results: List[SentimentResult], filename: str = "sentiment_results.csv"):
    """Save sentiment analysis results to CSV"""
    data = []
    for result in results:
        row = {
            "text": result.text,
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "reasoning": result.reasoning
        }
        # Add emotion scores if available
        if result.emotions:
            for emotion, score in result.emotions.items():
                row[f"emotion_{emotion}"] = score
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def analyze_from_file(file_path: str, text_column: str = "text") -> List[SentimentResult]:
    """Analyze sentiment from a CSV file"""
    try:
        df = pd.read_csv(file_path)
        texts = df[text_column].astype(str).tolist()
        
        analyzer = SentimentAnalyzer()
        results = analyzer.analyze_batch(texts)
        
        return results
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Example usage and testing
def main():
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()
    
    # Sample texts for testing
    sample_texts = [
        "I absolutely love this product! It exceeded all my expectations.",
        "This is the worst experience I've ever had. Completely disappointed.",
        "The weather is okay today, nothing special.",
        "I'm so excited about the upcoming vacation!",
        "I'm feeling really anxious about the presentation tomorrow.",
        "I'm feeling terrible since yesterday with all the horrendous activities happening",
        " This is a horrendous feeling and I am utterly disappointed with this"
    ]
    
    print("=== Simple Sentiment Analysis ===")
    for text in sample_texts[:3]:
        sentiment = analyzer.analyze_simple(text)
        print(f"Text: {text[:50]}...")
        print(f"Sentiment: {sentiment}\n")
    
    print("=== Detailed Sentiment Analysis ===")
    for text in sample_texts[:2]:
        result = analyzer.analyze_detailed(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result.sentiment} (confidence: {result.confidence})")
        print(f"Reasoning: {result.reasoning}")
        if result.emotions:
            print("Emotions:", {k: f"{v:.2f}" for k, v in result.emotions.items()})
        print("-" * 50)
    
    print("=== Contextual Analysis ===")
    context = "This is a review of a restaurant"
    text = "The service was slow but the food was amazing"
    result = analyzer.analyze_with_context(text, context)
    print(f"Context: {context}")
    print(f"Text: {text}")
    print(f"Sentiment: {result.sentiment} (confidence: {result.confidence})")
    print(f"Reasoning: {result.reasoning}")
    
    print("=== Batch Analysis ===")
    batch_results = analyzer.analyze_batch(sample_texts)
    save_results_to_csv(batch_results, "sample_sentiment_results.csv")
    
    # Print summary statistics
    sentiments = [r.sentiment for r in batch_results if r.sentiment != "error"]
    print(f"\nSummary:")
    print(f"Positive: {sentiments.count('positive')}")
    print(f"Negative: {sentiments.count('negative')}")
    print(f"Neutral: {sentiments.count('neutral')}")

# Advanced feature: Custom sentiment categories
class CustomSentimentAnalyzer(SentimentAnalyzer):
    """Extended analyzer with custom sentiment categories"""
    
    def analyze_custom_categories(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Analyze sentiment using custom categories"""
        categories_str = ", ".join(categories)
        
        prompt = f"""
        Analyze the sentiment of the following text using these specific categories: {categories_str}
        
        Text: "{text}"
        
        Provide scores (0-1) for each category. Respond with JSON format:
        {{
            "category1": 0.8,
            "category2": 0.2,
            ...
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a sentiment analyst. Provide scores for custom categories as valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error in custom analysis: {e}")
            return {cat: 0.0 for cat in categories}

# Example of custom categories usage
def custom_analysis_example():
    analyzer = CustomSentimentAnalyzer()
    
    text = "The new iPhone has amazing features but it's way too expensive"
    custom_categories = ["excitement", "frustration", "satisfaction", "disappointment"]
    
    scores = analyzer.analyze_custom_categories(text, custom_categories)
    print(f"Text: {text}")
    print("Custom sentiment scores:")
    for category, score in scores.items():
        print(f"  {category}: {score:.2f}")

if __name__ == "__main__":
    # Make sure to set your OpenAI API key
            main()
                    