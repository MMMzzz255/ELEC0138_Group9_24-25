from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import tempfile

app = Flask(__name__)

# 加载预训练模型
model = joblib.load('svm_model.pkl')  # 请确保模型文件存在


def preprocess_data(input_df):
    """预处理函数（与训练时保持一致）"""
    # 移除训练时删除的列
    if 'Label' in input_df.columns:
        input_df = input_df.drop(columns=['Label'])
    if 'Class' in input_df.columns:
        input_df = input_df.drop(columns=['Class'])
    return input_df


@app.route('/predict', methods=['POST'])
def predict():
    """处理CSV文件预测请求"""
    # 检查文件是否上传
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # 保存临时文件
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        # 读取CSV文件
        df = pd.read_csv(temp_path)

        # 预处理
        processed_df = preprocess_data(df)

        # 验证特征维度
        if processed_df.shape[1] != 78:  # 根据你的实际特征数量修改
            return jsonify({'error': f'Invalid feature dimension. Expected 78, got {processed_df.shape[1]}'}), 400

        # 预测
        predictions = model.predict(processed_df)

        # 转换预测结果为标签
        results = ['DDOS Attack' if pred == 1 else 'Normal' for pred in predictions]

        # 统计结果（修复 int64 序列化问题）
        attack_count = int(sum(predictions))  # 转换为 Python int
        normal_count = int(len(predictions) - attack_count)  # 转换为 Python int

        return jsonify({
            'predictions': results,
            'statistics': {
                'total_samples': len(results),
                'ddos_attacks': attack_count,
                'normal_traffic': normal_count,
                'attack_ratio': f"{attack_count / len(results):.2%}"
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


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