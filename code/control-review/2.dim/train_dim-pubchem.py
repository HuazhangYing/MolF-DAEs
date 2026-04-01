import os
import sys
import gc
import ctypes

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,5,6,7"
# =========================
# 0. 先读取命令行参数
# 用法:
# CUDA_VISIBLE_DEVICES=0 python train_dim.py 2
# CUDA_VISIBLE_DEVICES=1 python train_dim.py 3
# =========================
# dim = 3  # notebook里默认先手动指定
# print(f"Running latent dim = {dim}")

# =========================
# 1. 预加载 CUDA 11 库
# 注意：这一步必须在 import tensorflow 之前
# =========================
lib_path = "/home/yinghuazhang/miniconda3/envs/daes/lib"
libs = [
    "libcudart.so.11.0",
    "libcublas.so.11",
    "libcublasLt.so.11",
    "libcufft.so.10",
    "libcurand.so.10",
    "libcusolver.so.11",
    "libcusparse.so.11",
    "libcudnn.so.8",
]

print("--- 开始预加载 CUDA 11 库 ---")
for lib in libs:
    full_path = os.path.join(lib_path, lib)
    try:
        ctypes.CDLL(full_path)
        print(f"✅ {lib} 加载成功")
    except Exception as e:
        print(f"❌ {lib} 加载失败: {e}")

# =========================
# 2. 设置 GPU 可见性
# 最好不要在这里写死 1,5,6,7
# 而是由外部命令控制：
# CUDA_VISIBLE_DEVICES=0 python train_dim.py 2
# =========================
# 不在代码里强制写死单卡
# 由外部启动命令决定每个进程使用哪张GPU
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"))

# =========================
# 3. 再导入 TensorFlow
# =========================
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("Visible GPUs:", gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ 已成功为 {len(gpus)} 张显卡开启显存增长模式")
    except RuntimeError as e:
        print(f"⚠️ 设置显存增长时出错: {e}")

# =========================
# 4. 其余常规模块导入
# =========================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 脚本模式下保存图片，不依赖显示器
import matplotlib.pyplot as plt

from joblib import load, dump
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D,
    Flatten, UpSampling2D, Reshape
)
from tensorflow.keras.models import Sequential

from molmap.model import RegressionEstimator, MultiClassEstimator, MultiLabelEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from molmap import dataset, MolMap, feature


from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import tensorflow as tf
import gc

dim = int(sys.argv[1])
print("Training dim =", dim)
base_model_dir = "/data/yinghuazhang/MolF-DAEs/code/control-review/model"
base_result_dir = "/data/yinghuazhang/MolF-DAEs/result/pubchemfp"
data_dir = "/data/yinghuazhang/MolF-DAEs/dataset/pubchem_molecule3.data2"

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
 
    try:
        del classifier # this is from global space - change this as you need
    except:
        pass
 
    print(gc.collect()) # if it does something you should see a number as output
 
    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.compat.v1.Session(config=config))



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_history_plot(history_dict, save_path, title):
    df = pd.DataFrame(history_dict)
    ax = df.plot(figsize=(6, 4))
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_reconstruction_plot(x_true, x_pred, save_path, n_show=10):
    fig = plt.figure(figsize=(20, 8))
    for i in range(n_show):
        ax = plt.subplot(4, 5, i + 1)
        ax.imshow(x_true[i], cmap="gray")
        ax.set_title(f"True {i}")
        ax.axis("off")

        ax = plt.subplot(4, 5, i + 1 + n_show)
        ax.imshow(x_pred[i], cmap="gray")
        ax.set_title(f"Recon {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

np.random.seed(42)
X1 = load(data_dir)
idx = np.random.choice(X1.shape[0], 80000, replace=False)
X1 = X1[idx]

class Encoder(Model):
    def __init__(self, dim):
        super().__init__()
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(512, activation='relu')
        self.d3 = Dense(128, activation='relu')
        self.d4 = Dense(64, activation='relu')
        self.d5 = Dense(32, activation='relu')
        self.d6 = Dense(dim, activation='relu')
        
    def call(self,x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        return self.d6(x)       
    
    
class Decoder(Model):
    def __init__(self):
        super().__init__()
        self.d7 = Dense(32, activation='relu')
        self.d8 = Dense(64, activation='relu')
        self.d9 = Dense(128, activation='relu')
        self.d10 = Dense(512, activation='relu')
        self.d11 = Dense(1024, activation='relu')
        self.d12 = Dense(729, activation='sigmoid')
        self.re = Reshape((27,27))
    
    def call(self,x):
        x = self.d7(x)
        x = self.d8(x)
        x = self.d9(x)
        x = self.d10(x)
        x = self.d11(x)
        x = self.d12(x)
        return  self.re(x)

class Autoencoder(Model):
    def __init__(self, dim):
        super().__init__()
        self.encoder = Encoder(dim)
        self.decoder = Decoder()

    def call(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec
reset_keras()

# 建议先确保 X1 是 float32
X1 = X1.astype("float32")

ensure_dir(base_model_dir)
ensure_dir(base_result_dir)

save_dir = os.path.join(base_result_dir, f"test9-{dim}D")
fig_dir = os.path.join(save_dir, "figures")
ensure_dir(save_dir)
ensure_dir(fig_dir)

print(f"Start training dim={dim}")

model = Autoencoder(dim=dim)

# ===== Stage 1: BCE =====
model.compile(optimizer='adam', loss='binary_crossentropy')
history1 = model.fit(
    X1, X1,
    batch_size=768,
    epochs=100,
    verbose=1
)

save_history_plot(
    history1.history,
    os.path.join(fig_dir, f"dim{dim}_stage1_bce_loss.png"),
    title=f"Latent dim={dim}, Stage 1 BCE Loss"
)

# ===== Stage 2: MSE =====
model.compile(optimizer='adam', loss='mse')
history2 = model.fit(
    X1, X1,
    batch_size=768,
    epochs=100,
    verbose=1
)

save_history_plot(
    history2.history,
    os.path.join(fig_dir, f"dim{dim}_stage2_mse_loss.png"),
    title=f"Latent dim={dim}, Stage 2 MSE Loss"
)

# ===== Reconstruction examples =====
y_pre = model.predict(X1[:10], verbose=0)

save_reconstruction_plot(
    X1[:10],
    y_pre,
    os.path.join(fig_dir, f"dim{dim}_reconstruction_examples.png"),
    n_show=10
)
model_path = os.path.join(base_model_dir, f"pubchemfp_autoencoder_dim{dim}")
model.save(model_path, save_format="tf")
print(f"Saved model to: {model_path}")

# ===== Save latent =====
X_latent = model.encoder(X1).numpy()
latent_path = os.path.join(save_dir, "latent_vectors.joblib")
dump(X_latent, latent_path)
print(f"Saved latent vectors to: {latent_path}")

# ===== Save history csv =====
pd.DataFrame(history1.history).to_csv(
    os.path.join(save_dir, "stage1_bce_history.csv"), index=False
)
pd.DataFrame(history2.history).to_csv(
    os.path.join(save_dir, "stage2_mse_history.csv"), index=False
)

# ===== Save summary =====
summary_df = pd.DataFrame({
    "dim": [dim],
    "final_bce_loss": [history1.history["loss"][-1]],
    "final_mse_loss": [history2.history["loss"][-1]],
    "n_samples": [X1.shape[0]]
})
summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)

print(f"Finished dim={dim}")