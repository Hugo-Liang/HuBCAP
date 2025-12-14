import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
all_commits = pd.read_csv('./ase_dataset_sept_19_2021.csv')

#Separate by language, since the Java commits are missing some info which we will add later on.
py = all_commits[all_commits.PL == 'python']
java = all_commits[all_commits.PL == 'java']

#Java first: partition into train/val/test and check # of commits
print("Java VF vs NVF for train/val/test")
java_train = java[java.partition =="train"].drop_duplicates(subset='commit_id')
java_val = java[java.partition == "val"].drop_duplicates(subset='commit_id')
java_test = java[java.partition == "test"].drop_duplicates(subset='commit_id')


#Python: partition into train/val/test and check # of commits
print("Py VF vs NVF for train/val/test")
py_train = py[py.partition =="train"].drop_duplicates(subset='commit_id')
py_val = py[py.partition == "val"].drop_duplicates(subset='commit_id')
py_test = py[py.partition == "test"].drop_duplicates(subset='commit_id')


tokenizer = AutoTokenizer.from_pretrained("codebert-base")

# 配色方案
# #AAD4F8  浅蓝色
# #D55276  
CUSTOM_PALETTE = {
    "Java": {
        "train": "#AAD4F8", 
        "valid": "#F1A7B5",  
        "test": "#D55276"   
    },
    "Python": {
        "train": "#AAD4F8",  
        "valid": "#F1A7B5",  
        "test": "#D55276"   
    }
}

# 定义分词函数（默认使用空格分词）
def tokenize(text):
    tokenized_msg = tokenizer(
        str(text),
        add_special_tokens=False
    )
    # print(tokenized_msg)
    # exit()
    return tokenized_msg["input_ids"]

# ========================
# 1. 数据预处理
# ========================
def preprocess(df, language, subset_name):
    """为每个子集添加语言标识和子集类型"""
    df = df.copy()
    # 计算序列长度
    df["seq_length"] = df["msg"].apply(lambda x: len(tokenize(x)))
    # 添加元信息列
    df["language"] = language
    df["subset"] = subset_name
    return df[["language", "subset", "seq_length"]]  # 保留关键列

# 合并Java数据
java_dfs = [
    preprocess(java_train, "Java", "train"),
    preprocess(java_val, "Java", "val"),
    preprocess(java_test, "Java", "test")
]
combined_java = pd.concat(java_dfs)

# 合并Python数据
py_dfs = [
    preprocess(py_train, "Python", "train"),
    preprocess(py_val, "Python", "val"),
    preprocess(py_test, "Python", "test")
]
combined_py = pd.concat(py_dfs)


# ========================
# 2. 可视化（生成两幅图）
# ========================
def plot_language(lang_df, language):
    """绘制单个语言的箱线图"""
     # 提取配色方案
    color_dict = CUSTOM_PALETTE[language]
    palette = [color_dict["train"], color_dict["valid"], color_dict["test"]]
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=lang_df,
        x="subset",
        y="seq_length",
        order=["train", "val", "test"],  # 固定子集顺序
        palette=palette,                 # 配色方案
        showfliers=False,                 # 不显示异常值
        width=0.4
    )


    # plt.title(f"{language} Commit Message Length Distribution", fontsize=14)
    plt.xlabel("Subset Type", fontsize=12)
    plt.ylabel("Token Sequence Length", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{language}_msg_box.svg")

# 生成两幅独立的箱线图
plot_language(combined_java, "Java")
plot_language(combined_py, "Python")

# ========================
# 3. 统计分析
# ========================
def print_stats(df, language):
    print(f"\n【{language}】长度分布统计:")
    stats = df.groupby("subset")["seq_length"].describe(
        percentiles=[.5, .75, .9, .95, .99]
    )
    print(stats.round(1).to_markdown())

print_stats(combined_java, "Java")
print_stats(combined_py, "Python")

# ========================
# 4. 推荐max_length
# ========================
def recommend_max_length(df, coverage=0.95):
    return df.groupby("subset")["seq_length"].quantile(coverage).astype(int).to_dict()

print("\n推荐max_length（覆盖95%样本）:")
print("Java:", recommend_max_length(combined_java))
print("Python:", recommend_max_length(combined_py))

# 全局安全值建议
safe_length_java = combined_java["seq_length"].quantile(0.95)
safe_length_py = combined_py["seq_length"].quantile(0.95)
print(f"\n全局安全值（覆盖所有子集95%样本）:")
print(f"Java: {int(safe_length_java)}, py: {int(safe_length_py)}")