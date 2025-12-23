import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import roc_curve, auc

def get_kl_divergence(img_c, img_s):
    p, _ = np.histogram(img_c, bins=256, range=(0, 256), density=True)
    q, _ = np.histogram(img_s, bins=256, range=(0, 256), density=True)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return np.sum(p * np.log(p / q))

def get_pdh(image):
    # Chênh lệch pixel kề nhau
    diff = np.abs(image[:, 1:].astype(np.int16) - image[:, :-1].astype(np.int16))
    hist, _ = np.histogram(diff, bins=256, range=(0, 256))
    return hist

def rs_analysis_demo(image):
    # Mô phỏng tỷ lệ Regular (Rm) và Singular (Sm)
    h, w = image.shape
    flat = image.flatten()
    diffs = np.abs(flat[::2].astype(np.int16) - flat[1::2].astype(np.int16))
    rm = np.sum(diffs < 5)
    sm = len(diffs) - rm
    return rm/len(diffs), sm/len(diffs)

def plot_roc_demo():
    # Giả lập biểu đồ ROC cho báo cáo
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure("ROC Curve Analysis")
    plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
    plt.plot([0,1], [0,1], '--')
    plt.legend()
    plt.show()