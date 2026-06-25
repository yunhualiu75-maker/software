import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PATIENT_PHOTO_FOLDER'] = 'static/patient_photos'
app.config['PRESCRIPTION_FOLDER'] = 'prescriptions'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PATIENT_PHOTO_FOLDER'], exist_ok=True)
os.makedirs(app.config['PRESCRIPTION_FOLDER'], exist_ok=True)

# ================== 人脸识别模型 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_tensor)
    features = features.view(features.size(0), -1)
    return features.cpu().numpy()


def cosine_similarity(feat1, feat2):
    dot = (feat1 * feat2).sum()
    norm1 = (feat1 ** 2).sum() ** 0.5
    norm2 = (feat2 ** 2).sum() ** 0.5
    return dot / (norm1 * norm2)


# 全局数据
patient_df = None
patient_features = {}


def load_excel_data():
    global patient_df
    excel_path = None
    for ext in ['.xlsx', '.xls']:
        if os.path.exists(f'doctor{ext}'):
            excel_path = f'doctor{ext}'
            break
    if not excel_path:
        print("错误：未找到 doctor.xlsx 或 doctor.xls 文件，请将患者数据文件放在程序根目录下。")
        return False
    try:
        df = pd.read_excel(excel_path)
        required_cols = ['患者编号', '患者姓名', '性别', '年龄', '体重(kg)', '身高(cm)',
                         '家庭信息', '既往病历', '检查检验报告', '用药记录', '诊断历史']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Excel缺少必需列: {missing}")
            return False
        df['患者编号'] = df['患者编号'].astype(str)
        patient_df = df
        print(f"成功加载 {len(df)} 条患者记录")
        return True
    except Exception as e:
        print(f"加载Excel失败: {e}")
        return False


def find_patient_photo(patient_id):
    photo_dir = app.config['PATIENT_PHOTO_FOLDER']
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    for ext in extensions:
        photo_path = os.path.join(photo_dir, f"{patient_id}{ext}")
        if os.path.exists(photo_path):
            return photo_path
    return None


def load_patient_features():
    global patient_features
    patient_features = {}
    if patient_df is None:
        return
    for _, row in patient_df.iterrows():
        pid = str(row['患者编号'])
        photo_path = find_patient_photo(pid)
        if photo_path:
            try:
                feat = get_features(photo_path)
                patient_features[pid] = feat
                print(f"已加载患者 {pid} 人脸特征 ({os.path.basename(photo_path)})")
            except Exception as e:
                print(f"加载患者 {pid} 照片失败: {e}")
        else:
            print(f"警告：患者 {pid} 的照片不存在（支持 .jpg/.jpeg/.png/.bmp/.gif）")


# 启动时自动加载
with app.app_context():
    if load_excel_data():
        load_patient_features()
    else:
        print("请确保 doctor.xlsx 文件存在且格式正确，然后重启程序。")


# ================== Flask 路由 ==================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    if patient_df is None:
        return jsonify({'error': '患者数据未加载，请检查 doctor.xlsx'}), 400
    if not patient_features:
        return jsonify({'error': '未加载任何人脸特征，请确保照片文件存在且命名为 患者编号.扩展名'}), 400

    file = request.files.get('face_image')
    if not file:
        return jsonify({'error': '未上传图片'}), 400

    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_face.jpg')
    file.save(temp_path)

    try:
        input_feat = get_features(temp_path)
        best_match = None
        best_sim = -1
        threshold = 0.65

        for pid, feat in patient_features.items():
            sim = cosine_similarity(input_feat, feat)
            if sim > best_sim:
                best_sim = sim
                best_match = pid

        os.remove(temp_path)

        if best_match and best_sim >= threshold:
            patient_info = patient_df[patient_df['患者编号'] == best_match].iloc[0].to_dict()
            for k, v in patient_info.items():
                if pd.isna(v):
                    patient_info[k] = ''
                elif hasattr(v, 'item'):
                    patient_info[k] = v.item()
            patient_info['similarity'] = float(best_sim)
            return jsonify({'success': True, 'patient': patient_info})
        else:
            return jsonify({'success': False, 'message': f'未识别到患者，最高相似度 {best_sim:.3f} 低于阈值'})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500


@app.route('/get_patient_photo/<patient_id>')
def get_patient_photo(patient_id):
    """返回患者照片的URL（如果存在）"""
    photo_path = find_patient_photo(patient_id)
    if photo_path:
        # 转换为相对于 static 的 URL
        rel_path = os.path.relpath(photo_path, 'static')
        return jsonify({'photo_url': f'/static/{rel_path.replace(os.sep, "/")}'})
    else:
        return jsonify({'photo_url': None})


@app.route('/save_prescription', methods=['POST'])
def save_prescription():
    data = request.json
    patient_id = data.get('patient_id')
    patient_name = data.get('patient_name')
    items = data.get('items', [])

    if not patient_id or not items:
        return jsonify({'error': '缺少必要信息'}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prescription_{patient_id}_{timestamp}.txt"
    filepath = os.path.join(app.config['PRESCRIPTION_FOLDER'], filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"患者编号: {patient_id}\n")
        f.write(f"患者姓名: {patient_name}\n")
        f.write(f"开具时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")
        f.write("处 方 明 细\n")
        f.write("-" * 50 + "\n")
        for idx, item in enumerate(items, 1):
            f.write(f"{idx}. 药品: {item.get('medicine', '')}\n")
            f.write(f"   用法: {item.get('dosage', '')}\n")
            f.write(f"   备注: {item.get('note', '')}\n\n")

    return jsonify({'success': True, 'filename': filename, 'download_url': f'/download_prescription/{filename}'})


@app.route('/download_prescription/<filename>')
def download_prescription(filename):
    return send_from_directory(app.config['PRESCRIPTION_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)