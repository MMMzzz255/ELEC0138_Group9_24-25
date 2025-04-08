from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import tempfile
from datetime import datetime, timedelta

app = Flask(__name__)

# 加载模型
model = joblib.load('svm_model.pkl')  # 确保模型文件存在

# 防御参数
ATTACK_THRESHOLD = 10  # 攻击样本数量阈值
COOLDOWN_DURATION = timedelta(minutes=5)
cooldown_until = None  # 当前冷却结束时间

def preprocess_data(input_df):
    """预处理数据（与训练保持一致）"""
    if 'Label' in input_df.columns:
        input_df = input_df.drop(columns=['Label'])
    if 'Class' in input_df.columns:
        input_df = input_df.drop(columns=['Class'])
    return input_df

@app.before_request
def check_cooldown():
    """检查是否处于冷却期内"""
    global cooldown_until

    # 不限制 reset_defense 接口
    if request.path == '/reset_defense':
        return

    if cooldown_until and datetime.now() < cooldown_until:
        return jsonify({
            'error': 'Service is temporarily unavailable due to suspected DDoS attack. Please try again later.'
        }), 429


def trigger_defense_mechanism(attack_count, total_samples):
    """触发防御措施（限流 + 日志记录）"""
    global cooldown_until
    cooldown_until = datetime.now() + COOLDOWN_DURATION

    with open("ddos_defense_log.txt", "a") as log:
        log.write(f"[{datetime.now()}] DDoS Detected: {attack_count}/{total_samples} samples.\n")
        log.write(f"Service cooldown active until: {cooldown_until}\n\n")

    print(f"[!] DDoS Attack Detected: {attack_count}/{total_samples}")
    print("[!] Defense triggered: entering cooldown.")

@app.route('/predict', methods=['POST'])
def predict():
    """处理上传文件并进行预测"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        df = pd.read_csv(temp_path)
        processed_df = preprocess_data(df)

        if processed_df.shape[1] != 78:
            return jsonify({'error': f'Invalid feature dimension. Expected 78, got {processed_df.shape[1]}'}), 400

        predictions = model.predict(processed_df)
        results = ['DDOS Attack' if pred == 1 else 'Normal' for pred in predictions]

        attack_count = int(sum(predictions))
        normal_count = int(len(predictions) - attack_count)

        if attack_count >= ATTACK_THRESHOLD:
            trigger_defense_mechanism(attack_count, len(predictions))

        return jsonify({
            'predictions': results,
            'statistics': {
                'total_samples': len(results),
                'ddos_attacks': attack_count,
                'normal_traffic': normal_count,
                'attack_ratio': f"{attack_count / len(results):.2%}"
            },
            'defense_triggered': attack_count >= ATTACK_THRESHOLD
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

@app.route('/reset_defense', methods=['POST'])
def reset_defense():
    """手动重置防御冷却期 curl -X POST http://localhost:5000/reset_defense"""
    global cooldown_until
    cooldown_until = None

    with open("ddos_defense_log.txt", "a") as log:
        log.write(f"[{datetime.now()}] Defense reset manually via /reset_defense\n")

    print("[*] Defense reset manually.")
    return jsonify({'message': 'Defense cooldown reset. Service is now available.'})

@app.route('/')
def index():
    return """
    <h1>DDoS Detection API</h1>
    <p>Send POST request with CSV file to /predict endpoint</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Predict">
    </form>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
