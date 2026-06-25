import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import os
from datetime import datetime

# ================== 设备配置 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== 模型加载 ==================
print("正在加载人脸识别模型...")
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()
print("模型加载完成")

# ================== 图像预处理 ==================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_img(img_path):
    """预处理图像"""
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    return img.to(device)


def get_features(img_path):
    """提取图像特征"""
    img_tensor = preprocess_img(img_path)
    with torch.no_grad():
        features = model(img_tensor)
    features = features.view(features.size(0), -1)
    return features


def compute_similarity(feat1, feat2):
    """计算余弦相似度"""
    feat1 = feat1.cpu().numpy()
    feat2 = feat2.cpu().numpy()
    dot = (feat1 * feat2).sum()
    norm1 = (feat1 ** 2).sum() ** 0.5
    norm2 = (feat2 ** 2).sum() ** 0.5
    return dot / (norm1 * norm2)


# ================== 医生工作站主程序 ==================
class DoctorWorkstation:
    def __init__(self, root):
        self.root = root
        self.root.title("智能医生工作站 - 人脸识别病历系统")
        self.root.geometry("1400x800")

        # 数据存储
        self.patient_data = None  # 存储Excel数据
        self.current_patient = None  # 当前选中的患者
        self.face_features_cache = {}  # 缓存人脸特征 {patient_id: features}
        self.photo_dir = "patient_photos"  # 患者照片存放目录

        # 创建照片目录
        if not os.path.exists(self.photo_dir):
            os.makedirs(self.photo_dir)

        # 创建GUI界面
        self.create_widgets()

        # 摄像头相关
        self.cap = None
        self.is_capturing = False

    def create_widgets(self):
        """创建界面组件"""
        # 顶部工具栏
        toolbar = tk.Frame(self.root, bg="#2c3e50", height=60)
        toolbar.pack(fill=tk.X)

        tk.Button(toolbar, text="加载患者数据(Excel)", command=self.load_excel_data,
                  bg="#3498db", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=10, pady=10)

        tk.Button(toolbar, text="摄像头人脸识别", command=self.start_face_recognition,
                  bg="#27ae60", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=10, pady=10)

        tk.Button(toolbar, text="从照片识别", command=self.recognize_from_photo,
                  bg="#e67e22", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=10, pady=10)

        tk.Button(toolbar, text="保存当前处方", command=self.save_prescription,
                  bg="#9b59b6", fg="white", font=("Arial", 12), padx=10).pack(side=tk.LEFT, padx=10, pady=10)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪 - 请加载患者数据Excel文件")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 主内容区域 - 分割为左右两部分
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：患者列表
        left_frame = tk.Frame(main_pane)
        main_pane.add(left_frame, width=400)

        tk.Label(left_frame, text="患者列表", font=("Arial", 14, "bold")).pack(pady=5)

        # 患者列表树形视图
        columns = ("patient_id", "name", "gender", "age")
        self.patient_tree = ttk.Treeview(left_frame, columns=columns, show="headings", height=20)
        self.patient_tree.heading("patient_id", text="患者编号")
        self.patient_tree.heading("name", text="姓名")
        self.patient_tree.heading("gender", text="性别")
        self.patient_tree.heading("age", text="年龄")
        self.patient_tree.column("patient_id", width=100)
        self.patient_tree.column("name", width=100)
        self.patient_tree.column("gender", width=60)
        self.patient_tree.column("age", width=60)
        self.patient_tree.pack(fill=tk.BOTH, expand=True)

        # 绑定选择事件
        self.patient_tree.bind("<<TreeviewSelect>>", self.on_patient_select)

        # 右侧：病历详情
        right_frame = tk.Frame(main_pane)
        main_pane.add(right_frame, width=1000)

        # 患者基本信息卡片
        info_frame = tk.LabelFrame(right_frame, text="患者基本信息", font=("Arial", 12, "bold"))
        info_frame.pack(fill=tk.X, pady=5)

        self.info_text = tk.Text(info_frame, height=6, font=("Arial", 11), wrap=tk.WORD)
        self.info_text.pack(fill=tk.X, padx=5, pady=5)

        # 病历标签页
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # 既往病历标签页
        self.history_frame = tk.Frame(notebook)
        notebook.add(self.history_frame, text="既往病历")
        self.history_text = tk.Text(self.history_frame, wrap=tk.WORD, font=("Arial", 10))
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 检查检验报告标签页
        self.report_frame = tk.Frame(notebook)
        notebook.add(self.report_frame, text="检查检验报告")
        self.report_text = tk.Text(self.report_frame, wrap=tk.WORD, font=("Arial", 10))
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 用药记录标签页
        self.medication_frame = tk.Frame(notebook)
        notebook.add(self.medication_frame, text="用药记录")
        self.medication_text = tk.Text(self.medication_frame, wrap=tk.WORD, font=("Arial", 10))
        self.medication_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 诊断历史标签页
        self.diagnosis_frame = tk.Frame(notebook)
        notebook.add(self.diagnosis_frame, text="诊断历史")
        self.diagnosis_text = tk.Text(self.diagnosis_frame, wrap=tk.WORD, font=("Arial", 10))
        self.diagnosis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 开具处方标签页
        self.prescription_frame = tk.Frame(notebook)
        notebook.add(self.prescription_frame, text="开具处方")

        # 处方编辑区域
        prescription_edit_frame = tk.Frame(self.prescription_frame)
        prescription_edit_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(prescription_edit_frame, text="药品名称:", font=("Arial", 10)).pack(anchor=tk.W)
        self.medicine_name = tk.Entry(prescription_edit_frame, font=("Arial", 10), width=40)
        self.medicine_name.pack(anchor=tk.W, pady=2)

        tk.Label(prescription_edit_frame, text="用法用量:", font=("Arial", 10)).pack(anchor=tk.W)
        self.dosage = tk.Entry(prescription_edit_frame, font=("Arial", 10), width=40)
        self.dosage.pack(anchor=tk.W, pady=2)

        tk.Label(prescription_edit_frame, text="备注:", font=("Arial", 10)).pack(anchor=tk.W)
        self.prescription_note = tk.Text(prescription_edit_frame, height=4, width=50)
        self.prescription_note.pack(anchor=tk.W, pady=2)

        tk.Button(prescription_edit_frame, text="添加至处方", command=self.add_to_prescription,
                  bg="#3498db", fg="white").pack(anchor=tk.W, pady=5)

        tk.Label(prescription_edit_frame, text="当前处方:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.prescription_list = tk.Text(prescription_edit_frame, height=8, width=70)
        self.prescription_list.pack(anchor=tk.W, pady=2, fill=tk.BOTH, expand=True)

        # 人脸显示区域（右下角）
        face_frame = tk.LabelFrame(right_frame, text="人脸识别", font=("Arial", 10, "bold"))
        face_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        self.face_label = tk.Label(face_frame, text="未加载图片", width=30, height=15, bg="#ecf0f1")
        self.face_label.pack(padx=5, pady=5)

        self.recog_result_label = tk.Label(face_frame, text="", font=("Arial", 10), wraplength=250)
        self.recog_result_label.pack(padx=5, pady=5)

    def load_excel_data(self):
        """加载Excel患者数据"""
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if not file_path:
            return

        try:
            self.patient_data = pd.read_excel(file_path)
            self.status_var.set(f"已加载 {len(self.patient_data)} 条患者记录")

            # 清空列表
            for item in self.patient_tree.get_children():
                self.patient_tree.delete(item)

            # 填充患者列表
            for _, row in self.patient_data.iterrows():
                self.patient_tree.insert("", tk.END, values=(
                    row.get("患者编号", ""),
                    row.get("患者姓名", ""),
                    row.get("性别", ""),
                    row.get("年龄", "")
                ))

            # 预加载人脸特征
            self.preload_face_features()

        except Exception as e:
            messagebox.showerror("错误", f"加载Excel文件失败: {str(e)}")

    def preload_face_features(self):
        """预加载所有患者的人脸特征"""
        if self.patient_data is None:
            return

        self.status_var.set("正在预加载人脸特征...")
        self.root.update()

        for _, row in self.patient_data.iterrows():
            patient_id = str(row.get("患者编号", ""))
            photo_path = os.path.join(self.photo_dir, f"{patient_id}.jpg")

            # 检查是否存在患者照片
            if os.path.exists(photo_path):
                try:
                    features = get_features(photo_path)
                    self.face_features_cache[patient_id] = features
                except Exception as e:
                    print(f"加载患者 {patient_id} 照片失败: {e}")

        self.status_var.set(f"已预加载 {len(self.face_features_cache)} 个患者的人脸特征")

    def start_face_recognition(self):
        """启动摄像头进行人脸识别"""
        if self.patient_data is None:
            messagebox.showwarning("警告", "请先加载患者数据Excel文件")
            return

        # 创建识别窗口
        recog_window = tk.Toplevel(self.root)
        recog_window.title("摄像头人脸识别")
        recog_window.geometry("640x580")

        # 视频显示区域
        video_label = tk.Label(recog_window)
        video_label.pack()

        # 结果标签
        result_label = tk.Label(recog_window, text="请面对摄像头，按空格键拍照识别", font=("Arial", 12))
        result_label.pack(pady=10)

        # 启动摄像头
        cap = cv2.VideoCapture(0)

        def update_frame():
            ret, frame = cap.read()
            if ret:
                # 显示帧
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                video_label.imgtk = imgtk
                video_label.config(image=imgtk)

            if recog_window.winfo_exists():
                recog_window.after(10, update_frame)

        def capture_photo(event=None):
            """拍照并识别"""
            ret, frame = cap.read()
            if ret:
                # 保存临时照片
                temp_path = "temp_capture.jpg"
                cv2.imwrite(temp_path, frame)

                # 进行识别
                result_label.config(text="正在识别中...")
                recog_window.update()

                recognized_id = self.recognize_face(temp_path)

                if recognized_id:
                    # 找到对应患者
                    patient_row = self.patient_data[self.patient_data["患者编号"].astype(str) == str(recognized_id)]
                    if not patient_row.empty:
                        result_label.config(
                            text=f"识别成功！患者编号: {recognized_id}, 姓名: {patient_row.iloc[0].get('患者姓名', '')}")
                        self.select_patient_by_id(recognized_id)
                        recog_window.after(1500, recog_window.destroy)
                    else:
                        result_label.config(text=f"识别到患者编号{recognized_id}，但未在数据库中找到")
                else:
                    result_label.config(text="识别失败，请确保照片清晰且该患者已在系统中")

                # 删除临时照片
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        update_frame()
        recog_window.bind("<space>", capture_photo)

        # 窗口关闭时释放摄像头
        def on_close():
            cap.release()
            recog_window.destroy()

        recog_window.protocol("WM_DELETE_WINDOW", on_close)

    def recognize_from_photo(self):
        """从文件选择照片进行识别"""
        if self.patient_data is None:
            messagebox.showwarning("警告", "请先加载患者数据Excel文件")
            return

        photo_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not photo_path:
            return

        recognized_id = self.recognize_face(photo_path)

        if recognized_id:
            patient_row = self.patient_data[self.patient_data["患者编号"].astype(str) == str(recognized_id)]
            if not patient_row.empty:
                messagebox.showinfo("识别成功",
                                    f"识别到患者\n编号: {recognized_id}\n姓名: {patient_row.iloc[0].get('患者姓名', '')}")
                self.select_patient_by_id(recognized_id)
            else:
                messagebox.showwarning("识别结果", f"识别到患者编号{recognized_id}，但未在数据库中找到")
        else:
            messagebox.showwarning("识别失败", "未能识别出患者身份\n请确保照片清晰且该患者已在系统中")

    def recognize_face(self, img_path):
        """识别照片对应的患者ID"""
        try:
            # 提取待识别照片的特征
            input_features = get_features(img_path)

            best_match = None
            best_similarity = -1
            threshold = 0.7

            # 与所有缓存的患者特征进行比对
            for patient_id, cached_features in self.face_features_cache.items():
                similarity = compute_similarity(input_features, cached_features)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = patient_id

            # 显示识别结果图片
            self.show_image_in_label(img_path, self.face_label)
            self.recog_result_label.config(text=f"匹配相似度: {best_similarity:.4f}")

            if best_similarity >= threshold:
                return best_match
            return None

        except Exception as e:
            print(f"识别失败: {e}")
            return None

    def show_image_in_label(self, img_path, label):
        """在Label中显示图片"""
        try:
            img = Image.open(img_path)
            img = img.resize((250, 250))
            imgtk = ImageTk.PhotoImage(img)
            label.config(image=imgtk)
            label.image = imgtk
        except Exception as e:
            print(f"显示图片失败: {e}")

    def select_patient_by_id(self, patient_id):
        """根据患者编号选择患者"""
        for item in self.patient_tree.get_children():
            values = self.patient_tree.item(item)["values"]
            if values and str(values[0]) == str(patient_id):
                self.patient_tree.selection_set(item)
                self.patient_tree.see(item)
                self.on_patient_select()
                break

    def on_patient_select(self, event=None):
        """患者选中事件"""
        selection = self.patient_tree.selection()
        if not selection:
            return

        item = selection[0]
        values = self.patient_tree.item(item)["values"]
        if not values:
            return

        patient_id = values[0]
        patient_name = values[1]

        # 查找完整患者信息
        patient_row = self.patient_data[self.patient_data["患者编号"].astype(str) == str(patient_id)]
        if patient_row.empty:
            return

        self.current_patient = patient_row.iloc[0]

        # 更新基本信息显示
        self.update_basic_info()

        # 更新各病历标签页
        self.update_history_tab()
        self.update_report_tab()
        self.update_medication_tab()
        self.update_diagnosis_tab()

        # 清空处方列表
        self.prescription_list.delete(1.0, tk.END)

        # 尝试加载患者照片
        photo_path = os.path.join(self.photo_dir, f"{patient_id}.jpg")
        if os.path.exists(photo_path):
            self.show_image_in_label(photo_path, self.face_label)
            self.recog_result_label.config(text=f"已加载患者照片")
        else:
            self.face_label.config(image="", text="暂无照片")
            self.face_label.image = None
            self.recog_result_label.config(text="未找到患者照片，请将照片命名为 {患者编号}.jpg 放入 patient_photos 目录")

        self.status_var.set(f"当前患者: {patient_name} (编号: {patient_id})")

    def update_basic_info(self):
        """更新基本信息显示"""
        if self.current_patient is None:
            return

        info = f"""
【患者编号】{self.current_patient.get('患者编号', '')}
【姓    名】{self.current_patient.get('患者姓名', '')}
【性    别】{self.current_patient.get('性别', '')}
【年    龄】{self.current_patient.get('年龄', '')}
【家庭信息】{self.current_patient.get('家庭信息', '')}
        """
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)

    def update_history_tab(self):
        """更新既往病历标签页"""
        if self.current_patient is None:
            return
        content = self.current_patient.get('既往病历', '无记录')
        self.history_text.delete(1.0, tk.END)
        self.history_text.insert(1.0, str(content))

    def update_report_tab(self):
        """更新检查检验报告标签页"""
        if self.current_patient is None:
            return
        content = self.current_patient.get('检查检验报告', '无记录')
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(1.0, str(content))

    def update_medication_tab(self):
        """更新用药记录标签页"""
        if self.current_patient is None:
            return
        content = self.current_patient.get('用药记录', '无记录')
        self.medication_text.delete(1.0, tk.END)
        self.medication_text.insert(1.0, str(content))

    def update_diagnosis_tab(self):
        """更新诊断历史标签页"""
        if self.current_patient is None:
            return
        content = self.current_patient.get('诊断历史', '无记录')
        self.diagnosis_text.delete(1.0, tk.END)
        self.diagnosis_text.insert(1.0, str(content))

    def add_to_prescription(self):
        """添加药品到处方列表"""
        if self.current_patient is None:
            messagebox.showwarning("警告", "请先选择患者")
            return

        medicine = self.medicine_name.get().strip()
        dosage_info = self.dosage.get().strip()
        note = self.prescription_note.get(1.0, tk.END).strip()

        if not medicine:
            messagebox.showwarning("警告", "请输入药品名称")
            return

        prescription_line = f"【{medicine}】 {dosage_info} - {note}\n"
        self.prescription_list.insert(tk.END, prescription_line)

        # 清空输入
        self.medicine_name.delete(0, tk.END)
        self.dosage.delete(0, tk.END)
        self.prescription_note.delete(1.0, tk.END)

    def save_prescription(self):
        """保存处方"""
        if self.current_patient is None:
            messagebox.showwarning("警告", "请先选择患者")
            return

        prescription_content = self.prescription_list.get(1.0, tk.END).strip()
        if not prescription_content:
            messagebox.showwarning("警告", "处方为空")
            return

        # 保存到处方文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_id = self.current_patient.get('患者编号', 'unknown')
        filename = f"prescription_{patient_id}_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"患者编号: {self.current_patient.get('患者编号', '')}\n")
            f.write(f"患者姓名: {self.current_patient.get('患者姓名', '')}\n")
            f.write(f"开具时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            f.write(prescription_content)

        messagebox.showinfo("保存成功", f"处方已保存至: {filename}")
        self.status_var.set(f"处方已保存: {filename}")


# ================== 程序入口 ==================
if __name__ == "__main__":
    root = tk.Tk()
    app = DoctorWorkstation(root)
    root.mainloop()