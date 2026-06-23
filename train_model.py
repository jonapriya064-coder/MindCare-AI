"""
MindCare AI - Model Training Script
=====================================
Trains an emotion classification model using TF-IDF + Logistic Regression.
Run this script once before starting the app: python train_model.py
"""

import os
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ─────────────────────────────────────────────
# Training Data: (text, emotion) pairs
# Emotions: happy, sad, stress, angry, neutral
# ─────────────────────────────────────────────
TRAINING_DATA = [
    # ── HAPPY ──
    ("I feel so happy today", "happy"),
    ("Today was an amazing day", "happy"),
    ("I am so excited about my results", "happy"),
    ("Life is beautiful and I love it", "happy"),
    ("I got the job I wanted!", "happy"),
    ("My friends threw me a surprise party", "happy"),
    ("I feel joyful and grateful", "happy"),
    ("Everything is going great", "happy"),
    ("I feel blessed and wonderful", "happy"),
    ("I passed my exam with flying colors", "happy"),
    ("I am thrilled about the good news", "happy"),
    ("I feel positive and energetic today", "happy"),
    ("I love spending time with my family", "happy"),
    ("Today is a perfect day", "happy"),
    ("I feel content and peaceful", "happy"),
    ("I achieved my goal and I am overjoyed", "happy"),
    ("Feeling fantastic and full of energy", "happy"),
    ("My heart is full of happiness", "happy"),
    ("I can't stop smiling today", "happy"),
    ("Life feels great and I am grateful", "happy"),
    ("I feel cheerful and optimistic", "happy"),
    ("Today was absolutely wonderful", "happy"),
    ("I feel inspired and motivated", "happy"),
    ("I am enjoying every moment of life", "happy"),
    ("Good things keep happening to me", "happy"),

    # ── SAD ──
    ("I feel so sad and lonely", "sad"),
    ("Nobody cares about me", "sad"),
    ("I am heartbroken and devastated", "sad"),
    ("I lost someone I loved dearly", "sad"),
    ("Everything feels hopeless", "sad"),
    ("I have been crying all day", "sad"),
    ("I feel empty inside", "sad"),
    ("I miss my old friends", "sad"),
    ("I feel like giving up on everything", "sad"),
    ("My heart aches and I don't know why", "sad"),
    ("I feel disconnected from everyone", "sad"),
    ("Life doesn't feel worth living sometimes", "sad"),
    ("I feel broken and unloved", "sad"),
    ("I have no motivation to do anything", "sad"),
    ("I feel rejected and alone", "sad"),
    ("I am grieving and I can't stop crying", "sad"),
    ("Everything makes me feel sad", "sad"),
    ("I feel melancholy and hopeless", "sad"),
    ("I can't find joy in anything anymore", "sad"),
    ("I feel depressed and worthless", "sad"),
    ("My life feels like a failure", "sad"),
    ("I feel isolated and misunderstood", "sad"),
    ("I am overwhelmed with sadness", "sad"),
    ("I wish things were different", "sad"),
    ("I feel numb and emotionally drained", "sad"),

    # ── STRESS ──
    ("I am so stressed about my exams", "stress"),
    ("I have too much work and no time", "stress"),
    ("I feel overwhelmed by everything", "stress"),
    ("I cannot handle the pressure anymore", "stress"),
    ("My deadlines are killing me", "stress"),
    ("I feel anxious about the future", "stress"),
    ("I can't sleep because of all the stress", "stress"),
    ("I have been worrying nonstop", "stress"),
    ("I feel burnout from all the work", "stress"),
    ("Everything is piling up and I feel stuck", "stress"),
    ("I am stressed about my financial situation", "stress"),
    ("I feel like I am going to break down", "stress"),
    ("There is too much on my plate right now", "stress"),
    ("I cannot focus because I am so anxious", "stress"),
    ("I feel tense and wound up all the time", "stress"),
    ("My head is spinning with all these thoughts", "stress"),
    ("I cannot relax no matter what I do", "stress"),
    ("I am panicking about my presentation", "stress"),
    ("I feel nervous and uneasy all day", "stress"),
    ("Work pressure is making me lose my mind", "stress"),
    ("I have been feeling anxious and on edge", "stress"),
    ("I can't stop overthinking everything", "stress"),
    ("My mind won't stop racing at night", "stress"),
    ("I feel under constant pressure", "stress"),
    ("I am exhausted from all the stress", "stress"),

    # ── ANGRY ──
    ("I am so angry right now", "angry"),
    ("I feel furious and frustrated", "angry"),
    ("I cannot believe what they did to me", "angry"),
    ("I am boiling with rage", "angry"),
    ("People keep taking advantage of me", "angry"),
    ("I am fed up with everything", "angry"),
    ("I hate how unfair life is", "angry"),
    ("I want to scream because I am so angry", "angry"),
    ("I feel betrayed and enraged", "angry"),
    ("Everyone around me is so irritating", "angry"),
    ("I can't stand this situation anymore", "angry"),
    ("I feel resentful and bitter", "angry"),
    ("I am disgusted with how I was treated", "angry"),
    ("I feel like exploding with anger", "angry"),
    ("Why does everyone make me so mad", "angry"),
    ("I am livid about what happened", "angry"),
    ("I feel hostile and aggressive", "angry"),
    ("My anger is out of control", "angry"),
    ("I am infuriated by the injustice", "angry"),
    ("I feel indignant and outraged", "angry"),
    ("They made me so angry I can't think straight", "angry"),
    ("I am seething with frustration", "angry"),
    ("I feel like I want to break something", "angry"),
    ("I cannot calm down I am so upset", "angry"),
    ("My temper is through the roof today", "angry"),

    # ── NEUTRAL ──
    ("Today was just a regular day", "neutral"),
    ("Nothing special happened today", "neutral"),
    ("I feel okay, not great not bad", "neutral"),
    ("I had a normal day at work", "neutral"),
    ("I feel indifferent about things", "neutral"),
    ("Things are just going as usual", "neutral"),
    ("I don't feel anything particular right now", "neutral"),
    ("It was an average day", "neutral"),
    ("I am neither happy nor sad", "neutral"),
    ("I feel calm and undisturbed", "neutral"),
    ("Everything is the same as usual", "neutral"),
    ("I went through my routine today", "neutral"),
    ("Nothing exciting is happening in my life", "neutral"),
    ("I feel balanced and neutral", "neutral"),
    ("My day was uneventful", "neutral"),
    ("I feel fine, just ordinary", "neutral"),
    ("Things are stable and normal", "neutral"),
    ("I had a mediocre day", "neutral"),
    ("I feel neither stressed nor relaxed", "neutral"),
    ("Just another day in life", "neutral"),
    ("I am feeling okay today", "neutral"),
    ("My mood is stable and normal", "neutral"),
    ("Nothing much is going on", "neutral"),
    ("I feel alright and composed", "neutral"),
    ("It was a pretty standard day", "neutral"),
]


def clean_text(text):
    """Basic text cleaning: lowercase and remove special characters."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_and_save_model():
    """Train emotion classifier and save to disk."""
    print("=" * 50)
    print("  MindCare AI - Model Training")
    print("=" * 50)

    # Prepare data
    texts = [clean_text(t) for t, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]

    print(f"\n📊 Total training samples : {len(texts)}")
    print(f"📌 Emotion classes        : {sorted(set(labels))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Build Pipeline: TF-IDF → Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),   # Unigrams + bigrams
            max_features=5000,
            stop_words='english',
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            multi_class='auto',
            random_state=42
        ))
    ])

    # Train
    print("\n🔄 Training model ...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs('model', exist_ok=True)
    model_path = os.path.join('model', 'sentiment_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    print("=" * 50)
    print("✅ Training complete! You can now run: python app.py")
    print("=" * 50)


if __name__ == '__main__':
    train_and_save_model()
