"""
MindCare AI - Main Flask Application
======================================
An intelligent mental health assistant for educational wellness support.

DISCLAIMER: This is NOT a medical diagnosis tool. Always seek professional help.

Run:
    python train_model.py   # First time only
    python app.py
"""

import os
import re
import json
import joblib
import random
import sqlite3
from datetime import datetime, timedelta
from functools import wraps

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify)
from werkzeug.security import generate_password_hash, check_password_hash

from config import Config
from database import get_db, init_db, get_random_quote

# ── App Setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)

# ── Load ML Model ──────────────────────────────────────────────────────────
model = None
def load_model():
    global model
    if os.path.exists(Config.MODEL_PATH):
        model = joblib.load(Config.MODEL_PATH)
        print("✅ ML model loaded.")
    else:
        print("⚠️  Model not found. Run: python train_model.py")

# ── Emotion Metadata ───────────────────────────────────────────────────────
EMOTION_DATA = {
    'happy': {
        'emoji': '😊',
        'color': '#28a745',
        'label': 'Happy',
        'description': 'You seem to be in a great mood! Keep spreading positivity.',
    },
    'sad': {
        'emoji': '😢',
        'color': '#6c757d',
        'label': 'Sad',
        'description': 'It\'s okay to feel sad. Allow yourself to process these emotions.',
    },
    'stress': {
        'emoji': '😰',
        'color': '#fd7e14',
        'label': 'Stressed',
        'description': 'You seem stressed. Let\'s work on bringing some calm into your day.',
    },
    'angry': {
        'emoji': '😠',
        'color': '#dc3545',
        'label': 'Angry',
        'description': 'Anger is a natural emotion. Let\'s find healthy ways to manage it.',
    },
    'neutral': {
        'emoji': '😐',
        'color': '#17a2b8',
        'label': 'Neutral',
        'description': 'You seem balanced and calm. A good state to be in!',
    },
}

# ── Wellness Recommendations ───────────────────────────────────────────────
RECOMMENDATIONS = {
    'happy': [
        {'icon': '🎵', 'title': 'Share Your Joy', 'desc': 'Call a friend and share this positive energy.'},
        {'icon': '📝', 'title': 'Gratitude Journal', 'desc': 'Write down 3 things you are grateful for today.'},
        {'icon': '🎯', 'title': 'Set a Goal', 'desc': 'Use this positive energy to set a new personal goal.'},
        {'icon': '🌟', 'title': 'Help Someone', 'desc': 'Spread your happiness by doing a kind act today.'},
        {'icon': '🏃', 'title': 'Stay Active', 'desc': 'Channel your energy into a fun physical activity.'},
    ],
    'sad': [
        {'icon': '🫂', 'title': 'Reach Out', 'desc': 'Talk to a trusted friend or family member about how you feel.'},
        {'icon': '🚶', 'title': 'Take a Walk', 'desc': 'A 10-minute walk in nature can lift your spirits.'},
        {'icon': '🎵', 'title': 'Uplifting Music', 'desc': 'Listen to your favorite music to boost your mood.'},
        {'icon': '📖', 'title': 'Self-Compassion', 'desc': 'Be kind to yourself. Sadness is a valid emotion.'},
        {'icon': '🛁', 'title': 'Self-Care', 'desc': 'Take a warm bath or do something soothing for yourself.'},
        {'icon': '🩺', 'title': 'Seek Support', 'desc': 'If sadness persists, please talk to a mental health professional.'},
    ],
    'stress': [
        {'icon': '🧘', 'title': 'Deep Breathing', 'desc': 'Try 4-7-8 breathing: Inhale 4s, hold 7s, exhale 8s.'},
        {'icon': '📋', 'title': 'Prioritize Tasks', 'desc': 'Make a list and tackle one thing at a time.'},
        {'icon': '😴', 'title': 'Rest Well', 'desc': 'Ensure you get 7-8 hours of sleep tonight.'},
        {'icon': '🏋️', 'title': 'Exercise', 'desc': 'Even 20 minutes of exercise reduces stress hormones.'},
        {'icon': '📵', 'title': 'Digital Detox', 'desc': 'Take a 30-minute break from all screens.'},
        {'icon': '🍵', 'title': 'Herbal Tea', 'desc': 'A warm cup of chamomile or green tea can calm nerves.'},
    ],
    'angry': [
        {'icon': '🧊', 'title': 'Cool Down', 'desc': 'Give yourself a 10-minute break before reacting.'},
        {'icon': '🏃', 'title': 'Physical Release', 'desc': 'Go for a run or do some jumping jacks to release tension.'},
        {'icon': '✍️', 'title': 'Write It Out', 'desc': 'Write your feelings in a journal to process your anger.'},
        {'icon': '🌬️', 'title': 'Box Breathing', 'desc': 'Breathe in 4s, hold 4s, out 4s, hold 4s. Repeat.'},
        {'icon': '🗣️', 'title': 'Talk Calmly', 'desc': 'When calm, express your feelings using "I" statements.'},
        {'icon': '🎨', 'title': 'Creative Outlet', 'desc': 'Channel your anger into art, music, or writing.'},
    ],
    'neutral': [
        {'icon': '🧘', 'title': 'Mindfulness', 'desc': 'Practice 5 minutes of mindfulness meditation.'},
        {'icon': '📚', 'title': 'Learn Something', 'desc': 'Use this calm state to read or learn a new skill.'},
        {'icon': '🌿', 'title': 'Nature Time', 'desc': 'Spend some time outdoors to refresh your mind.'},
        {'icon': '📝', 'title': 'Plan Ahead', 'desc': 'Use this balanced state to plan your upcoming week.'},
        {'icon': '🤝', 'title': 'Social Connection', 'desc': 'Reach out to someone you haven\'t spoken to in a while.'},
    ],
}

# ── Chatbot Responses ──────────────────────────────────────────────────────
CHATBOT_RESPONSES = {
    'lonely': [
        "I hear you — feeling lonely can be really hard. You're not alone in feeling this way. 💙",
        "Loneliness is tough. Have you tried reaching out to an old friend today? Sometimes a simple message makes a big difference.",
        "It's okay to feel lonely sometimes. What's one small thing you could do to connect with someone today?",
    ],
    'fail': [
        "Failure is part of growth, not the end of your story. Every successful person has failed many times before. 💪",
        "I'm sorry to hear that. Remember — one setback doesn't define your entire journey. What can you learn from this experience?",
        "Failing doesn't make you a failure. It makes you someone who tried. Take a breath and try again when you're ready.",
    ],
    'sleep': [
        "Sleep trouble is really common, especially under stress. Try a consistent bedtime, no screens an hour before bed, and deep breathing. 😴",
        "Poor sleep can affect everything. Would you like to try a simple 4-7-8 breathing exercise before bed tonight?",
        "Your mind may be too active to rest. Try writing down your thoughts before sleep to 'empty' your mind onto paper.",
    ],
    'anxious': [
        "Anxiety can feel overwhelming. Try to ground yourself — name 5 things you can see, 4 you can touch, 3 you can hear. 🌿",
        "I understand anxiety can be really challenging. Deep breathing is a great tool — try inhaling for 4 counts and exhaling for 8 counts.",
        "Anxiety is your body's alarm system misfiring. Remind yourself: you are safe, right here, right now.",
    ],
    'happy': [
        "That's wonderful to hear! 🌟 What made your day special? Savor this feeling — you deserve it!",
        "Yay! Happiness is contagious. Keep doing whatever is bringing you joy today! 😊",
        "That's so great! Consider writing this moment down in your journal so you can revisit it on harder days.",
    ],
    'stress': [
        "Stress can feel crushing, but it's manageable. Have you tried the box breathing technique? Breathe in 4s, hold 4s, out 4s. 🧘",
        "I understand you're under pressure. Let's break it down — what's the ONE most important thing you need to handle right now?",
        "Feeling overwhelmed is a sign you need a break. Even 5 minutes of calm can reset your nervous system.",
    ],
    'sad': [
        "I'm really sorry you're feeling sad. Your feelings are completely valid. 💙 Is there someone you trust you could talk to?",
        "Sadness is part of being human. Allow yourself to feel it without judgment. Things will get better.",
        "I hear you. On difficult days, try to be extra gentle with yourself. Small comforts matter.",
    ],
    'angry': [
        "It's okay to feel angry — it's a valid emotion. Try to give yourself some space before reacting. 🌬️",
        "Anger often signals that something important to us has been threatened. What do you think is really bothering you underneath the anger?",
        "Take a few deep breaths. When you're ready, try to express what you're feeling calmly using 'I feel' statements.",
    ],
    'worthless': [
        "Please know that you have inherent worth and value. Those negative thoughts are not facts. 💙",
        "Those feelings of worthlessness can be very painful. You matter more than you realize. Would you consider speaking to a counselor?",
        "You are not worthless. Depression and low moods can create very distorted thoughts. Please reach out to someone who can help.",
    ],
    'help': [
        "I'm here to support you. 💙 For immediate help, please contact a mental health helpline in your area.",
        "It's brave to ask for help. Please consider speaking to a therapist, counselor, or trusted person in your life.",
        "You deserve support. I'm an AI and have limits, but a mental health professional can provide real help. Please reach out to one.",
    ],
    'default': [
        "Thank you for sharing that with me. 💙 How long have you been feeling this way?",
        "I hear you. It takes courage to talk about how you're feeling. What would make things feel a little better right now?",
        "I understand. Remember — every emotion is temporary, and you have the strength to get through this.",
        "I'm here to listen. Would you like to try one of our wellness exercises or check your mood history?",
        "That sounds really difficult. Remember, it's okay to not be okay sometimes. Please don't hesitate to seek professional support if needed.",
    ],
}

def get_chatbot_response(user_message):
    """Return a relevant chatbot response based on keywords in the message."""
    msg = user_message.lower()
    for keyword, responses in CHATBOT_RESPONSES.items():
        if keyword in msg:
            return random.choice(responses)
    return random.choice(CHATBOT_RESPONSES['default'])

# ── Auth Decorator ─────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def clean_text(text):
    """Clean input text for ML model."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

# ── Landing Page ───────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

# ── Register ───────────────────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        # Validation
        if not name or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')

        conn = get_db()
        try:
            conn.execute(
                'INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                (name, email, generate_password_hash(password))
            )
            conn.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already registered. Please log in.', 'danger')
        finally:
            conn.close()

    return render_template('register.html')

# ── Login ──────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        conn = get_db()
        user = conn.execute(
            'SELECT * FROM users WHERE email = ?', (email,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['negative_count'] = 0
            flash(f'Welcome back, {user["name"]}! 👋', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')

# ── Logout ─────────────────────────────────────────────────────────────────
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index'))

# ── Dashboard ──────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db()
    user_id = session['user_id']

    # Recent moods (last 5)
    recent_moods = conn.execute(
        'SELECT * FROM moods WHERE user_id = ? ORDER BY date DESC LIMIT 5',
        (user_id,)
    ).fetchall()

    # Today's mood count
    today = datetime.now().strftime('%Y-%m-%d')
    today_count = conn.execute(
        "SELECT COUNT(*) FROM moods WHERE user_id = ? AND date(date) = ?",
        (user_id, today)
    ).fetchone()[0]

    # Total entries
    total = conn.execute(
        'SELECT COUNT(*) FROM moods WHERE user_id = ?', (user_id,)
    ).fetchone()[0]

    # Journal count
    journals = conn.execute(
        'SELECT COUNT(*) FROM journal_entries WHERE user_id = ?', (user_id,)
    ).fetchone()[0]

    conn.close()

    quote = get_random_quote()
    now = datetime.now()

    return render_template('dashboard.html',
        recent_moods=recent_moods,
        today_count=today_count,
        total_entries=total,
        journal_count=journals,
        quote=quote,
        now=now,
        emotion_data=EMOTION_DATA,
    )

# ── Mood Analysis ──────────────────────────────────────────────────────────
@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    return render_template('analyze.html')

@app.route('/analyze/result', methods=['POST'])
@login_required
def analyze_result():
    text = request.form.get('text', '').strip()
    save_mood = request.form.get('save_mood') == 'on'
    journal_text = request.form.get('journal', '').strip()

    if not text or len(text) < 3:
        flash('Please enter at least a few words to analyze.', 'warning')
        return redirect(url_for('analyze'))

    # Predict emotion
    if model:
        cleaned = clean_text(text)
        emotion = model.predict([cleaned])[0]
        proba = model.predict_proba([cleaned])[0]
        classes = model.classes_.tolist()
        confidence = round(max(proba) * 100, 1)
        all_probs = {c: round(p * 100, 1) for c, p in zip(classes, proba)}
    else:
        emotion = 'neutral'
        confidence = 85.0
        all_probs = {'neutral': 85.0, 'happy': 5.0, 'sad': 5.0, 'stress': 3.0, 'angry': 2.0}

    # Track negative moods for emergency warning
    negative_emotions = ['sad', 'angry', 'stress']
    if emotion in negative_emotions:
        session['negative_count'] = session.get('negative_count', 0) + 1
    else:
        session['negative_count'] = 0

    # Save mood to database if requested
    if save_mood:
        conn = get_db()
        conn.execute(
            'INSERT INTO moods (user_id, emotion, journal) VALUES (?, ?, ?)',
            (session['user_id'], emotion, journal_text or text)
        )
        conn.commit()
        conn.close()
        flash('Mood saved to your history! 📝', 'success')

    emotion_info = EMOTION_DATA.get(emotion, EMOTION_DATA['neutral'])
    recs = RECOMMENDATIONS.get(emotion, RECOMMENDATIONS['neutral'])
    show_emergency = session.get('negative_count', 0) >= Config.NEGATIVE_MOOD_THRESHOLD

    return render_template('result.html',
        text=text,
        emotion=emotion,
        emotion_info=emotion_info,
        confidence=confidence,
        all_probs=all_probs,
        recommendations=recs,
        show_emergency=show_emergency,
    )

# ── Chatbot ────────────────────────────────────────────────────────────────
@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html')

@app.route('/chatbot/message', methods=['POST'])
@login_required
def chatbot_message():
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'response': 'Please type a message.'})
    response = get_chatbot_response(user_msg)
    return jsonify({'response': response})

# ── Mood History ───────────────────────────────────────────────────────────
@app.route('/history')
@login_required
def history():
    conn = get_db()
    moods = conn.execute(
        'SELECT * FROM moods WHERE user_id = ? ORDER BY date DESC',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    return render_template('history.html', moods=moods, emotion_data=EMOTION_DATA)

@app.route('/history/delete/<int:mood_id>', methods=['POST'])
@login_required
def delete_mood(mood_id):
    conn = get_db()
    conn.execute(
        'DELETE FROM moods WHERE id = ? AND user_id = ?',
        (mood_id, session['user_id'])
    )
    conn.commit()
    conn.close()
    flash('Mood entry deleted.', 'info')
    return redirect(url_for('history'))

# ── Journal ────────────────────────────────────────────────────────────────
@app.route('/journal', methods=['GET', 'POST'])
@login_required
def journal():
    conn = get_db()
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        text = request.form.get('text', '').strip()
        if text:
            conn.execute(
                'INSERT INTO journal_entries (user_id, title, text) VALUES (?, ?, ?)',
                (session['user_id'], title, text)
            )
            conn.commit()
            flash('Journal entry saved! ✍️', 'success')

    entries = conn.execute(
        'SELECT * FROM journal_entries WHERE user_id = ? ORDER BY date DESC',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    return render_template('journal.html', entries=entries)

@app.route('/journal/delete/<int:entry_id>', methods=['POST'])
@login_required
def delete_journal(entry_id):
    conn = get_db()
    conn.execute(
        'DELETE FROM journal_entries WHERE id = ? AND user_id = ?',
        (entry_id, session['user_id'])
    )
    conn.commit()
    conn.close()
    flash('Journal entry deleted.', 'info')
    return redirect(url_for('journal'))

# ── Analytics ──────────────────────────────────────────────────────────────
@app.route('/analytics')
@login_required
def analytics():
    conn = get_db()
    user_id = session['user_id']

    # Emotion distribution (all time)
    distribution = conn.execute(
        '''SELECT emotion, COUNT(*) as count FROM moods
           WHERE user_id = ? GROUP BY emotion''',
        (user_id,)
    ).fetchall()

    # Last 7 days
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    weekly = conn.execute(
        '''SELECT date(date) as day, emotion, COUNT(*) as count
           FROM moods WHERE user_id = ? AND date(date) >= ?
           GROUP BY day, emotion ORDER BY day''',
        (user_id, seven_days_ago)
    ).fetchall()

    # Last 30 days
    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    monthly = conn.execute(
        '''SELECT date(date) as day, emotion, COUNT(*) as count
           FROM moods WHERE user_id = ? AND date(date) >= ?
           GROUP BY day ORDER BY day''',
        (user_id, thirty_days_ago)
    ).fetchall()

    conn.close()

    # Build chart data
    emotion_counts = {e: 0 for e in EMOTION_DATA}
    for row in distribution:
        emotion_counts[row['emotion']] = row['count']

    return render_template('analytics.html',
        emotion_counts=emotion_counts,
        weekly=weekly,
        monthly=monthly,
        emotion_data=EMOTION_DATA,
    )

# ── Recommendations ────────────────────────────────────────────────────────
@app.route('/recommendations')
@login_required
def recommendations():
    conn = get_db()
    last_mood = conn.execute(
        'SELECT emotion FROM moods WHERE user_id = ? ORDER BY date DESC LIMIT 1',
        (session['user_id'],)
    ).fetchone()
    conn.close()

    emotion = last_mood['emotion'] if last_mood else 'neutral'
    recs = RECOMMENDATIONS.get(emotion, RECOMMENDATIONS['neutral'])
    emotion_info = EMOTION_DATA.get(emotion, EMOTION_DATA['neutral'])

    return render_template('recommendation.html',
        recommendations=recs,
        emotion=emotion,
        emotion_info=emotion_info,
        all_recs=RECOMMENDATIONS,
        emotion_data=EMOTION_DATA,
    )

# ── Emergency Support ──────────────────────────────────────────────────────
@app.route('/emergency')
def emergency():
    return render_template('emergency.html')

# ── Error Handlers ─────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('404.html'), 500

# ── App Startup ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    load_model()
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)
