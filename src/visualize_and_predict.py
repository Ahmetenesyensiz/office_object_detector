import os
import torch
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from torchvision import transforms
from custom_cnn import get_model
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import random
import cv2

# Sınıf isimlerini güncelle
CLASS_NAMES = ['bardak', 'bilgisayar', 'canta', 'defter', 'fare', 'kalem', 'klavye', 'telefon']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(device)

def get_latest_model():
    model_dir = "models"
    if not os.path.exists(model_dir):
        return None
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return None
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    return os.path.join(model_dir, latest_model)

model_path = get_latest_model()
if model_path:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

def get_random_image():
    """Veri setinden rastgele bir görsel seçer"""
    data_dir = "data/new_test_images"  # Yeni görsellerin bulunduğu klasör
    if not os.path.exists(data_dir):
        messagebox.showerror("Hata", f"Veri seti bulunamadı: {data_dir}")
        return None, None
        
    # Tüm görselleri listele
    all_images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not all_images:
        messagebox.showerror("Hata", "Veri setinde görsel bulunamadı!")
        return None, None
    
    # Rastgele bir görsel seç
    selected_image = random.choice(all_images)
    return selected_image, None  # Gerçek sınıf bilinmediği için None döndürüyoruz

def predict_image(image_path):
    """Görsel için tahmin yapar"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return image, CLASS_NAMES[predicted_class], confidence, probabilities[0].cpu().numpy()

def predict_random_image():
    """Rastgele bir görsel seçip tahmin yapar"""
    image_path, _ = get_random_image()  # Gerçek sınıf bilgisi artık kullanılmıyor
    if image_path is None:
        return

    image, predicted_class, confidence, probs = predict_image(image_path)
    
    # Sonuçları göster
    result_text = f"Tahmin: {predicted_class}\n"
    result_text += f"Güven: {confidence:.2%}"
    result_label.config(text=result_text)

    # Görseli göster
    img_resized = image.resize((200, 200))
    tk_img = ImageTk.PhotoImage(img_resized)
    image_panel.config(image=tk_img)
    image_panel.image = tk_img

    # Bar chart olarak olasılıkları göster
    plt.figure(figsize=(8, 6))
    plt.bar(CLASS_NAMES, probs)
    plt.xticks(rotation=45)
    plt.title("Sınıf Olasılıkları")
    plt.tight_layout()
    plt.show()

def predict_selected_image():
    """Kullanıcının seçtiği görsel için tahmin yapar"""
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    image, predicted_class, confidence, probs = predict_image(file_path)
    
    # Sonuçları göster
    result_text = f"Tahmin: {predicted_class}\n"
    result_text += f"Güven: {confidence:.2%}"
    result_label.config(text=result_text)

    # Görseli göster
    img_resized = image.resize((200, 200))
    tk_img = ImageTk.PhotoImage(img_resized)
    image_panel.config(image=tk_img)
    image_panel.image = tk_img

    # Bar chart olarak olasılıkları göster
    plt.figure(figsize=(8, 6))
    plt.bar(CLASS_NAMES, probs)
    plt.xticks(rotation=45)
    plt.title("Sınıf Olasılıkları")
    plt.tight_layout()
    plt.show()

def predict_from_frame(frame):
    """Webcam frame'inden tahmin yapar"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # OpenCV BGR formatından RGB'ye dönüştür
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return CLASS_NAMES[predicted_class], confidence

def start_webcam():
    """Webcam'i başlatır ve gerçek zamanlı tahmin yapar"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        messagebox.showerror("Hata", "Webcam başlatılamadı!")
        return

    def update_frame():
        ret, frame = cap.read()
        if ret:
            # Tahmin yap
            predicted_class, confidence = predict_from_frame(frame)
            
            # Sonucu göster
            result_text = f"Tahmin: {predicted_class}\n"
            result_text += f"Güven: {confidence:.2%}"
            result_label.config(text=result_text)

            # Frame'i göster
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (200, 200))
            img = Image.fromarray(frame_resized)
            tk_img = ImageTk.PhotoImage(image=img)
            image_panel.config(image=tk_img)
            image_panel.image = tk_img
            
            # Her 100ms'de bir güncelle
            root.after(100, update_frame)
        else:
            cap.release()

    update_frame()

# Ana pencere
root = tk.Tk()
root.title("Ofis Nesnesi Sınıflandırıcı")
root.geometry("500x400")

# Butonlar
btn_frame = ttk.Frame(root)
btn_frame.pack(pady=20)

btn_random = ttk.Button(btn_frame, text="Rastgele Görsel Seç", command=predict_random_image)
btn_random.pack(side=tk.LEFT, padx=10)

btn_select = ttk.Button(btn_frame, text="Görsel Seç", command=predict_selected_image)
btn_select.pack(side=tk.LEFT, padx=10)

btn_webcam = ttk.Button(btn_frame, text="Webcam Başlat", command=start_webcam)
btn_webcam.pack(side=tk.LEFT, padx=10)

# Sonuç etiketi
result_label = ttk.Label(root, text="Henüz bir tahmin yapılmadı", font=("Helvetica", 12))
result_label.pack(pady=10)

# Görsel paneli
image_panel = tk.Label(root)
image_panel.pack()

root.mainloop()
