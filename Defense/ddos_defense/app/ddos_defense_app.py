from flask import Flask, request, jsonify, redirect, url_for, session
import pandas as pd
import joblib
import os
import tempfile
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)

if not os.path.exists('../defense_log'):
    os.makedirs('../defense_log')

try:
    model = joblib.load('../model/rf_model.pkl')
    scaler = joblib.load('../model/rf_scaler.pkl')
    imputer = joblib.load('../model/rf_imputer.pkl')
    top_features = joblib.load('../model/rf_top_features.pkl')
except FileNotFoundError as e:
    raise RuntimeError(f"Required model artifact not found: {e.filename}")

# Defense parameters
ATTACK_RATIO_THRESHOLD = 0.2
COOLDOWN_DURATION = timedelta(minutes=5)
cooldown_until = None


def preprocess_data(input_df):
    input_df = input_df[top_features]
    X_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    X_scaled = pd.DataFrame(scaler.transform(X_imputed), columns=X_imputed.columns)
    return X_scaled


@app.before_request
def check_cooldown():
    global cooldown_until
    public_routes = ['/', '/reset_defense', '/favicon.ico']
    if request.path in public_routes:
        return
    if cooldown_until and datetime.now() < cooldown_until:
        return jsonify({'error': 'Service temporarily unavailable'}), 503


@app.route('/')
def index():
    global cooldown_until
    now = datetime.now()
    reset_success = request.args.get('reset_success')
    analysis_done = request.args.get('analysis_done')
    safe_stats = session.pop('safe_stats', None)
    attack_stats = session.pop('attack_stats', None)
    error_msg = session.pop('upload_error', None)

    if cooldown_until and now < cooldown_until:
        remaining = cooldown_until - now
        return (
            f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Service Temporarily Unavailable</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #fff3f3;
            text-align: center;
            padding: 50px;
        }}
        h1 {{
            color: #e74c3c;
        }}
        p {{
            font-size: 18px;
        }}
        button {{
            margin-top: 20px;
            padding: 10px 20px;
            font-size: 16px;
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <h1>Service Temporarily Unavailable</h1>
    <p>DDoS attack detected. Auto-defense enabled.</p>
    <p>Estimated recovery time: {cooldown_until.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Remaining time: {remaining.seconds // 3600}h {(remaining.seconds // 60) % 60}m</p>
    <form action='/reset_defense' method='post'>
        <button type='submit'>Disable Defense</button>
    </form>
    <script>setTimeout(()=>location.reload(),30000);</script>
</body>
</html>''', 503
        )

    result_message = ''
    if safe_stats and analysis_done:
        result_message = f'''
        <div class="safe">
            <h2>Analysis Complete</h2>
            <p>Total Samples: {safe_stats['total_samples']}</p>
            <p>Attack Count: {safe_stats['attack_count']}</p>
            <p class="safe-message">{safe_stats['message']}</p>
            <p class="timestamp">Analysis time: {datetime.fromisoformat(safe_stats['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        '''
    elif attack_stats and analysis_done:
        result_message = f'''
        <div class="error">
            <h2> DDoS Attack Detected</h2>
            <p>Total Samples: {attack_stats['total_samples']}</p>
            <p>Attack Count: {attack_stats['attack_count']}</p>
            <p class="safe-message"> Attack Ratio: {attack_stats['attack_ratio']}</p>
        </div>
        '''

    error_message = f'<div class="error">{error_msg}</div>' if error_msg else ''

    return (
        f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DDoS Detection System</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #f4f6f8;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 50px;
        }}
        h1 {{
            color: #2c3e50;
        }}
        form {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}
        input[type='file'], input[type='submit'] {{
            display: block;
            margin: 10px 0;
            font-size: 16px;
        }}
        .success {{
            color: green;
            margin-bottom: 10px;
        }}
        .error {{
            color: #e74c3c;
            margin-bottom: 10px;
            padding: 10px;
            background: #ffebee;
            border-radius: 4px;
        }}
        .note {{
            margin-top: 15px;
            color: #555;
        }}
        .safe {{
            background: #e8f5e9;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            text-align: center;
        }}
        .safe h2 {{
            color: #2e7d32;
            margin-top: 0;
        }}
        .safe-message {{
            font-weight: bold;
            font-size: 1.2em;
            color: #43a047;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            margin-top: 8px;
        }}
    </style>
</head>
<body>
    {f'<div class="success">Defense status has been reset</div>' if reset_success else ''}
    {error_message}
    {result_message}
    <h1>DDoS Traffic Detection System</h1>
    <form action='/predict' method='post' enctype='multipart/form-data'>
        <label>Upload CSV file:</label>
        <input type='file' name='file' required>
        <input type='submit' value='Start Analysis'>
    </form>
    <div class='note'>CSV format only. Requires Top 8 network traffic features.</div>
</body>
</html>'''
    )


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        session['upload_error'] = 'No file uploaded'
        return redirect(url_for('index'))

    file = request.files['file']
    if not file.filename.lower().endswith('.csv'):
        session['upload_error'] = 'Only accept CSV'
        return redirect(url_for('index'))

    try:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)

        df = pd.read_csv(temp_path)
        processed_df = preprocess_data(df)

        if processed_df.shape[1] != len(top_features):
            session['upload_error'] = f'Invalid feature dimension. Expected {len(top_features)}, got {processed_df.shape[1]}'
            return redirect(url_for('index'))

        predictions = model.predict(processed_df)
        attack_count = int(np.sum(predictions))
        total_samples = int(len(predictions))
        attack_ratio = attack_count / total_samples

        log_entry = f"[{datetime.now()}] Attack Detected\nTotal Samples: {total_samples}\nAttack Count: {attack_count}\nAttack Ratio: {attack_ratio:.2%}"
        with open("../defense_log/defense.log", "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        global cooldown_until
        if attack_ratio >= ATTACK_RATIO_THRESHOLD:
            cooldown_until = datetime.now() + COOLDOWN_DURATION
            session['attack_stats'] = {
                'total_samples': total_samples,
                'attack_count': attack_count,
                'attack_ratio': f"{attack_ratio:.2%}"
            }
        else:
            session['safe_stats'] = {
                'timestamp': datetime.now().isoformat(),
                'total_samples': total_samples,
                'attack_count': attack_count,
                'message': 'No DDoS attack detected'
            }

        return redirect(url_for('index') + "?analysis_done=true")

    except pd.errors.EmptyDataError:
        session['upload_error'] = 'Empty CSV file'
        return redirect(url_for('index'))
    except Exception as e:
        session['upload_error'] = f'Processing failed: {str(e)}'
        return redirect(url_for('index'))
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            os.rmdir(temp_dir)


@app.route('/reset_defense', methods=['POST'])
def reset_defense():
    global cooldown_until
    cooldown_until = None
    session.pop('attack_stats', None)
    return redirect(url_for('index') + "?reset_success=true")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
