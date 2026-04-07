import os
import yaml

# 从data.yaml加载类别信息
def load_classes_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

# 统计单个文件夹中各类别数量
def count_classes_in_dir(label_dir, classes):
    class_count = {cls: 0 for cls in classes}
    if not os.path.exists(label_dir):
        print(f"警告：目录 {label_dir} 不存在")
        return class_count
    
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(label_dir, filename), 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            cls_idx = int(line.split()[0])
                            if 0 <= cls_idx < len(classes):
                                cls_name = classes[cls_idx]
                                class_count[cls_name] += 1
                        except (ValueError, IndexError):
                            print(f"警告：文件 {filename} 中的行 '{line}' 格式不正确，已跳过")
    return class_count

# 主函数
def main():
    # 配置路径
    yaml_path = "data.yaml"  # data.yaml文件路径
    label_dirs = {
        '训练集': 'labels/train',
        '验证集': 'labels/val',
        '测试集': 'labels/test'
    }
    
    # 加载类别名称
    classes = load_classes_from_yaml(yaml_path)
    print(f"类别列表：{classes}\n")
    
    # 统计每个文件夹
    all_counts = {}
    for set_name, dir_path in label_dirs.items():
        counts = count_classes_in_dir(dir_path, classes)
        all_counts[set_name] = counts
        print(f"====={set_name}类别数量=====")
        for cls, cnt in counts.items():
            print(f"{cls}: {cnt}")
        print()  # 空行分隔
    
    # 计算总数量
    total_counts = {cls: 0 for cls in classes}
    for counts in all_counts.values():
        for cls, cnt in counts.items():
            total_counts[cls] += cnt
    
    print("=====总类别数量=====")
    for cls, cnt in total_counts.items():
        print(f"{cls}: {cnt}")

if __name__ == "__main__":
    main()