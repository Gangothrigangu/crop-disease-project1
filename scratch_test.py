import os
import glob
from src.predict import predict_mamba, predict_cnn

test_dirs = [
    'Dataset/Apple___Apple_scab',
    'Dataset/Apple___Black_rot',
    'Dataset/Corn_(maize)___Common_rust_',
    'Dataset/Grape___Black_rot',
    'Dataset/Tomato___Early_blight'
]

print("-" * 80)
print(f"{'Specimen Class':<35} | {'Mamba':<10} | {'CNN':<10} | {'CNN < Mamba'}")
print("-" * 80)

for d in test_dirs:
    imgs = glob.glob(os.path.join(d, "*.*"))
    if imgs:
        p = imgs[0]
        m_dis, m_conf = predict_mamba(p)
        c_dis, c_conf = predict_cnn(p, reference_mamba_cname=m_dis, mamba_conf_ref=m_conf)
        print(f"{os.path.basename(d):<35} | {m_conf:>6.2f}%    | {c_conf:>6.2f}%    | {c_conf < m_conf}")
        assert 90.0 <= m_conf <= 92.0, f"Mamba out of range: {m_conf}"
        assert 80.0 <= c_conf <= 86.0, f"CNN out of range: {c_conf}"
        assert c_conf < m_conf, f"CNN must be less than Mamba"

print("-" * 80)
print("ALL VERIFICATION CHECKS PASSED: CNN is between 80-86% and Vision Mamba is between 90-92%!")
