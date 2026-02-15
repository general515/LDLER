import os
import pandas as pd

# ===== 1. 根据你自己的路径改这里 =====
DATASET_ROOT = r"H:/datasets/DAiSEE"          # DAiSEE 总目录
FRAME_ROOT   = os.path.join(DATASET_ROOT, "DataSet")
LABEL_ROOT   = os.path.join(DATASET_ROOT, "Labels")

train_frames_dir = os.path.join(FRAME_ROOT, "TrainFrames")
val_frames_dir   = os.path.join(FRAME_ROOT, "ValidationFrames")

train_label_csv  = os.path.join(LABEL_ROOT, "TrainLabels.csv")
val_label_csv    = os.path.join(LABEL_ROOT, "ValidationLabels.csv")

# 输出位置（放在项目下的 csv 目录）
os.makedirs("csv", exist_ok=True)
train_out_csv = os.path.join("csv", "train.csv")
val_out_csv   = os.path.join("csv", "validation.csv")
# =====================================


def build_map(label_csv_path):
    """从官方标签文件中构建: ClipID -> Engagement 映射"""
    df = pd.read_csv(label_csv_path)

    # 你截图中的列名就是这几个：
    # ClipID, Boredom, Engagement, Confusion, Frustration
    assert "ClipID" in df.columns and "Engagement" in df.columns

    id2eng = {}
    for clip, eng in zip(df["ClipID"], df["Engagement"]):
        # clip 形如 '4000221001.avi'
        id2eng[str(clip).strip()] = int(eng)

    return id2eng


def build_csv_for_split(frames_dir, label_map, out_csv_path):
    """遍历 *Frames 目录，生成 path,label 的 csv"""
    rows = []

    for subject in sorted(os.listdir(frames_dir)):
        subject_dir = os.path.join(frames_dir, subject)
        if not os.path.isdir(subject_dir):
            continue

        for video in sorted(os.listdir(subject_dir)):
            video_dir = os.path.join(subject_dir, video)
            if not os.path.isdir(video_dir):
                continue

            # 用文件夹名 video 构造 ClipID
            clip_id_avi = video + ".avi"
            clip_id_mp4 = video + ".mp4"

            if clip_id_avi in label_map:
                label = label_map[clip_id_avi]
            elif clip_id_mp4 in label_map:
                label = label_map[clip_id_mp4]
            else:
                print("⚠ 找不到标签，跳过：", video_dir)
                continue

            # 写 path（这里保存绝对路径，斜杠统一成 /）
            path = video_dir.replace("\\", "/") + "/"
            rows.append({"path": path, "label": label})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv_path, index=False)
    print(f"✅ 写入 {out_csv_path}，共 {len(rows)} 条样本")


if __name__ == "__main__":
    # 1) 读取标签映射
    train_map = build_map(train_label_csv)
    val_map   = build_map(val_label_csv)

    # 2) 生成 train.csv & validation.csv
    build_csv_for_split(train_frames_dir, train_map, train_out_csv)
    build_csv_for_split(val_frames_dir,   val_map,   val_out_csv)
