import json, asyncio
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# import nltk, uvicorn
from fastapi import FastAPI, HTTPException
from typing import List, Dict
from dataclasses import asdict, dataclass
import numpy as np
from statistics import mean
import modal

import nltk

nltk.data.path.append("/root/nltk_data")
# nltk.download("averaged_perceptron_tagger_eng")
# nltk.download("punkt")
# nltk.download("vader_lexicon")
# nltk.download("brown")


# NLTK stuff for analysis (import before downloads)
from nltk import sent_tokenize, word_tokenize, pos_tag, RegexpParser
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import brown
from nltk.probability import FreqDist

# Use your custom Dockerfile to build the Modal image
image = modal.Image.debian_slim().from_dockerfile("Dockerfile")
app = modal.App(name="text-analysis-api", image=image)
fastapi_app = FastAPI()

# CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class TextRequest(BaseModel):
    text: str

# Data models
@dataclass
class SentenceMetrics:
    text: str
    syntactic_depth: int
    word_count: int
    character_count: int
    average_word_length: float
    rarity_score: float
    sentiment_scores: Dict[str, float]

@dataclass
class TextAnalysis:
    text: str
    sentences: List[SentenceMetrics]
    average_depth: float
    complexity_score: float
    sentiment_summary: Dict[str, float]
    readability_metrics: Dict[str, float]

class TextAnalyzer:
    """Service class for analyzing text complexity."""
    def __init__(self):
        self.chunker = RegexpParser(r"""
            NP: {<DT>?<JJ>*<NN>}
            P: {<IN>}
            V: {<V.*>}
            PP: {<P> <NP>}
            VP: {<V> <NP|PP>*}
        """)
        self.sia = SentimentIntensityAnalyzer()
        self.word_freq = FreqDist(word.lower() for word in brown.words())
    
    def get_tree_depth(self, tree) -> int:
        if isinstance(tree, nltk.Tree):
            return 1 + max((self.get_tree_depth(child) for child in tree), default=0)
        return 0
    
    def calculate_word_rarity(self, word: str) -> float:
        count = self.word_freq[word.lower()]
        return 1 / (count + 1)
    
    def analyze_sentence(self, sentence: str) -> SentenceMetrics:
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        tree = self.chunker.parse(tagged)
        
        words = [w.lower() for w in tokens]
        word_lengths = [len(w) for w in words]
        rarity_scores = [self.calculate_word_rarity(w) for w in words]
        sentiment = self.sia.polarity_scores(sentence)
        
        return SentenceMetrics(
            text=sentence,
            syntactic_depth=self.get_tree_depth(tree),
            word_count=len(words),
            character_count=len(sentence),
            average_word_length=mean(word_lengths) if word_lengths else 0,
            rarity_score=mean(rarity_scores) if rarity_scores else 0,
            sentiment_scores=sentiment
        )
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        sentence_count = len(sentences)
        word_count = len(words)
        syllable_count = sum(self._count_syllables(word) for word in words)
        if sentence_count == 0 or word_count == 0:
            return {'flesch_reading_ease': 0, 'gunning_fog': 0, 'average_sentence_length': 0}
        return {
            'flesch_reading_ease': 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count),
            'gunning_fog': 0.4 * ((word_count / sentence_count) + 100 * (self._count_complex_words(words) / word_count)),
            'average_sentence_length': word_count / sentence_count
        }
    
    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word and word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        return max(1, count)
    
    def _count_complex_words(self, words: List[str]) -> int:
        return sum(1 for w in words if self._count_syllables(w) >= 3)
    
    def analyze_text(self, text: str) -> TextAnalysis:
        sentences = sent_tokenize(text)
        sentence_metrics = [self.analyze_sentence(sent) for sent in sentences]
        depths = [m.syntactic_depth for m in sentence_metrics]
        sentiments = [m.sentiment_scores['compound'] for m in sentence_metrics if 'compound' in m.sentiment_scores]
        avg_depth = mean(depths) if depths else 0
        avg_sentiment = mean(sentiments) if sentiments else 0
        complexity_factors = []
        if depths:
            complexity_factors.append(avg_depth / 10)
        if sentence_metrics:
            rarity_vals = [m.rarity_score for m in sentence_metrics]
            complexity_factors.append((mean(rarity_vals) * 100) if rarity_vals else 0)
            avg_word_len = [m.average_word_length for m in sentence_metrics]
            complexity_factors.append((mean(avg_word_len) / 5) if avg_word_len else 0)
        complexity_score = mean(complexity_factors) if complexity_factors else 0
        readability = self.calculate_readability(text)
        return TextAnalysis(
            text=text,
            sentences=sentence_metrics,
            average_depth=avg_depth,
            complexity_score=complexity_score,
            sentiment_summary={
                'average_sentiment': avg_sentiment,
                'sentiment_variance': float(np.var(sentiments)) if len(sentiments) > 1 else 0.0
            },
            readability_metrics=readability
        )

@fastapi_app.post("/extract-analyze-complexity")
def analyze_complexity(req: TextRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    analyzer = TextAnalyzer()
    analysis = analyzer.analyze_text(text)
    return {
        "text": analysis.text,
        "complexity_score": analysis.complexity_score,
        "readability": analysis.readability_metrics,
        "sentences": [
            {
                "text": s.text,
                "metrics": {
                    "syntactic_depth": s.syntactic_depth,
                    "word_count": s.word_count,
                    "character_count": s.character_count,
                    "average_word_length": s.average_word_length,
                    "rarity_score": s.rarity_score,
                    "sentiment": s.sentiment_scores
                }
            }
            for s in analysis.sentences
        ],
        "summary": {
            "average_depth": analysis.average_depth,
            "sentiment": analysis.sentiment_summary
        }
    }

class TextRequest(BaseModel):
    text: str

@fastapi_app.post("/extract-analyze-complexity-stream")
async def analyze_complexity_stream(req: TextRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    analyzer = TextAnalyzer()
    async def stream_analysis():
        try:
            yield json.dumps({"type": "start-tool", "content": "textAnalysis"}) + "\n"
            await asyncio.sleep(0)
            sentences = sent_tokenize(text)
            for idx, sentence in enumerate(sentences):
                metrics = analyzer.analyze_sentence(sentence)
                chunk_data = {
                    "type": "analysis-delta",
                    "index": idx + 1,
                    "content": {
                        "text": sentence,
                        "syntactic_depth": metrics.syntactic_depth,
                        "word_count": metrics.word_count,
                        "character_count": metrics.character_count,
                        "average_word_length": metrics.average_word_length,
                        "rarity_score": metrics.rarity_score,
                        "sentiment_scores": metrics.sentiment_scores
                    }
                }
                yield json.dumps(chunk_data) + "\n"
                await asyncio.sleep(0)
            metrics_obj = analyzer.analyze_text(text)
            summary_data = {
                "type": "analysis-summary",
                "content": asdict(metrics_obj)
            }
            yield json.dumps(summary_data) + "\n"
            await asyncio.sleep(0)
        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"
        yield json.dumps({"type": "finish", "content": ""}) + "\n"
        await asyncio.sleep(0)
    return StreamingResponse(stream_analysis(), media_type="application/json")

@fastapi_app.get("/")
def home():
    return {
        "message": "Text Complexity Analyzer -- active!",
        "endpoints": {
            "analyze_text_complexity": "/analyze-complexity"
        }
    }
    
@app.function()
@modal.asgi_app()
def fastapi():
    return fastapi_app
