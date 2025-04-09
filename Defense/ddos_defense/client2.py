from flask import Flask, request, jsonify, redirect, url_for, session
import pandas as pd
import joblib
import os
import tempfile
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Session encryption key

# Create defense log directory
if not os.path.exists('defense'):
    os.makedirs('defense')

# Load model
try:
    model = joblib.load('svm_model.pkl')
except FileNotFoundError:
    raise RuntimeError("Model file svm_model.pkl not found")

# Defense parameters
ATTACK_THRESHOLD = 10
COOLDOWN_DURATION = timedelta(minutes=5)
cooldown_until = None


def preprocess_data(input_df):
    """Preprocess input data"""
    columns_to_drop = ['Label', 'Class']
    existing_columns = [col for col in columns_to_drop if col in input_df.columns]
    return input_df.drop(columns=existing_columns)


@app.before_request
def check_cooldown():
    """Check cooldown status"""
    global cooldown_until

    public_routes = ['/', '/reset_defense', '/favicon.ico']
    if request.path in public_routes:
        return

    if cooldown_until and datetime.now() < cooldown_until:
        return jsonify({'error': 'Service temporarily unavailable'}), 503


@app.route('/')
def index():
    """Main interface"""
    global cooldown_until
    now = datetime.now()
    attack_stats = session.get('attack_stats')

    # Defense active page
    if cooldown_until and now < cooldown_until:
        remaining = cooldown_until - now
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Service Unavailable</title>
            <style>
                body {{ 
                    text-align: center; 
                    padding: 50px; 
                    font-family: Arial, sans-serif;
                    background-color: #f8d7da;
                }}
                .stats-box {{
                    background: white;
                    padding: 20px;
                    margin: 20px auto;
                    width: 60%;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .countdown {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px auto;
                    width: 60%;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .reset-btn {{
                    background: #28a745;
                    color: white;
                    padding: 12px 24px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }}
            </style>
        </head>
        <body>
            <div class="alert-icon">🚨</div>
            <h1>Service Temporarily Unavailable</h1>

            <div class="stats-box">
                <h3>Attack Statistics</h3>
                <p>Total Samples: {attack_stats['total_samples'] if attack_stats else 'N/A'}</p>
                <p>Attack Count: {attack_stats['attack_count'] if attack_stats else 'N/A'}</p>
                <p>Attack Ratio: {attack_stats['attack_ratio'] if attack_stats else 'N/A'}</p>
            </div>

            <div class="countdown">
                <p>DDoS attack detected, automatic defense activated</p>
                <p>Estimated recovery time: {cooldown_until.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Remaining time: {remaining.seconds // 3600} hours {(remaining.seconds // 60) % 60} minutes</p>
            </div>

            <form action="/reset_defense" method="post">
                <button type="submit" class="reset-btn">
                    🛡️ Disable Defense Now
                </button>
            </form>
            <script>setTimeout(()=>location.reload(),30000);</script>
        </body>
        </html>
        """, 503

    # Normal interface
    reset_success = request.args.get('reset_success')
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DDoS Detection System</title>
        <style>
            body {{ 
                text-align: center; 
                padding: 50px; 
                font-family: Arial, sans-serif;
                background-color: #f0f8ff;
            }}
            .upload-box {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 20px auto;
                width: 400px;
            }}
            .success-msg {{
                background: #d4edda;
                color: #155724;
                padding: 15px;
                margin: 20px auto;
                width: 60%;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        {'<div class="success-msg">✅ Defense status has been reset</div>' if reset_success else ''}
        <h1>DDoS Traffic Detection System</h1>
        <div class="upload-box">
            <form action="/predict" method="post" enctype="multipart/form-data">
                <h3>Upload Traffic Data</h3>
                <input type="file" name="file" required>
                <br>
                <input type="submit" value="Start Analysis" 
                    style="padding:12px 24px; background:#007bff; color:white; border:none; border-radius:5px; cursor:pointer;">
            </form>
        </div>
        <p>Supports CSV format, requires 78 network traffic features</p>
    </body>
    </html>
    """


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400

    try:
        # Save temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)

        # Data processing
        df = pd.read_csv(temp_path)
        processed_df = preprocess_data(df)

        if processed_df.shape[1] != 78:
            return jsonify({'error': f'Invalid feature dimension. Expected 78, got {processed_df.shape[1]}'}), 400

        # Prediction
        predictions = model.predict(processed_df)
        attack_count = int(np.sum(predictions))  # Convert to native int
        total_samples = int(len(predictions))  # Convert to native int
        attack_ratio = attack_count / total_samples

        # Terminal output
        print(f"""
        [Attack Report]
        Total Samples: {total_samples}
        Attack Count: {attack_count}
        Attack Ratio: {attack_ratio:.2%}
        """)

        # Write to log
        log_entry = f"""[{datetime.now()}] Attack Detected
        Total Samples: {total_samples}
        Attack Count: {attack_count}
        Attack Ratio: {attack_ratio:.2%}
        """
        with open("defense/defense.log", "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        # Trigger defense
        if attack_count >= ATTACK_THRESHOLD:
            global cooldown_until
            cooldown_until = datetime.now() + COOLDOWN_DURATION
            session['attack_stats'] = {
                'total_samples': total_samples,
                'attack_count': attack_count,
                'attack_ratio': f"{attack_ratio:.2%}"
            }
            return redirect(url_for('index'))

        return jsonify({
            'attack_count': attack_count,
            'total_samples': total_samples,
            'attack_ratio': f"{attack_ratio:.2%}"
        })

    except pd.errors.EmptyDataError:
        return jsonify({'error': 'Empty CSV file'}), 400
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            os.rmdir(temp_dir)


@app.route('/reset_defense', methods=['POST'])
def reset_defense():
    """Reset defense status"""
    global cooldown_until
    cooldown_until = None
    return redirect(url_for('index') + "?reset_success=true")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)