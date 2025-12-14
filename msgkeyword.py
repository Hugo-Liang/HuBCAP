import pandas as pd 
import re

if __name__ == "__main__":
    # 加载数据
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
    # 定义两个正则表达式（根据实际需求修改）
    REGEX_PATTERNS = {
        "strong": "(?i)(denial .o f .service |\bXX E\b |remote.code.execution|\bopen.redirect |OSVDB|\bvuln|\bCVE\b |\bXSS\b |\bReDoS\b |\bNVD\b |malicious |x-frame-options |attack |cross.site |exploit |directory. traversal |\bRCE\b |\bdos\b |\bXSRF \b |clickjack |session.f ixation|hijack |advisory|insecure |security |\bcross-origin\b |unauthori[z|s]ed |in f inite.loop)",  
        "medium": "(?i)(authenticat (e |ion)|brute f orce |bypass |constant .time |crack |credential |\bDoS\b|expos (e |ing)|hack |harden|injection|lockout |over f low |password |\bPoC\b |proo f .o f .concept |poison|privelage |\b(in)?secur (e |ity)|(de )?serializ|spoo f |timing|traversal )"  
    }

    def enhanced_count_matches(df, text_col="msg", label_col="label"):
        """带Label分布的增强统计"""
        df[text_col] = df[text_col].fillna('').astype(str)
        # df[label_col] = df[label_col].fillna(0).astype(int)  # 处理缺失标签
        # 计算匹配掩码
        mask_bug = df[text_col].str.contains(REGEX_PATTERNS["strong"], 
                                                flags=re.IGNORECASE, regex=True)
        mask_feature = df[text_col].str.contains(REGEX_PATTERNS["medium"], 
                                                    flags=re.IGNORECASE, regex=True)
        mask_both = mask_bug & mask_feature
        mask_any = mask_bug | mask_feature
        mask_unmatched = ~mask_any
    
        # 统计函数
        def get_stats(mask):
            subset = df[mask]
            total = subset.shape[0]
            label_1 = subset[label_col].sum()
            label_0 = total - label_1
            return total, label_1, label_0
        
        # 生成统计字典
        stats = {
            "Total": df.shape[0],
            "Strong": get_stats(mask_bug),
            "Medium": get_stats(mask_feature),
            "Both": get_stats(mask_both),
            "Unmatched": get_stats(mask_unmatched)
        }
        
        return stats

    def create_analysis_report(datasets, language):
        """生成多维分析报告"""
        report = []
        for split_name, df in datasets.items():
            stats = enhanced_count_matches(df)
            report.append({
                "Language": language,
                "Split": split_name.upper(),
                "Total": stats["Total"],
                "Strong_Total": stats["Strong"][0],
                "Strong_Label1": stats["Strong"][1],
                "Strong_Label0": stats["Strong"][2],
                "Medium_Total": stats["Medium"][0],
                "Medium_Label1": stats["Medium"][1],
                "Medium_Label0": stats["Medium"][2],
                "Both_Total": stats["Both"][0],
                "Both_Label1": stats["Both"][1],
                "Both_Label0": stats["Both"][2],
                "Unmatched_Total": stats["Unmatched"][0],
                "Unmatched_Label1": stats["Unmatched"][1],
                "Unmatched_Label0": stats["Unmatched"][2]
            })
        
        return pd.DataFrame(report)

    # 生成Java报告
    java_report = create_analysis_report({
        "train": java_train,
        "test": java_test,
        "val": java_val
    }, "Java")

    # 生成Python报告
    py_report = create_analysis_report({
        "train": py_train,
        "test": py_test,
        "val": py_val
    }, "Python")

    # 合并报告并添加汇总
    full_report = pd.concat([java_report, py_report], ignore_index=True)

    # 添加语言级汇总
    language_summary = full_report.groupby("Language").agg({
        "Total": "sum",
        "Strong_Total": "sum",
        "Strong_Label1": "sum",
        "Strong_Label0": "sum",
        "Medium_Total": "sum",
        "Medium_Label1": "sum",
        "Medium_Label0": "sum",
        "Both_Total": "sum",
        "Both_Label1": "sum",
        "Both_Label0": "sum",
        "Unmatched_Total": "sum",
        "Unmatched_Label1": "sum",
        "Unmatched_Label0": "sum"
    }).reset_index()
    language_summary["Split"] = "TOTAL"

    # 最终报告格式处理
    final_report = pd.concat([full_report, language_summary], ignore_index=True)
    final_report.sort_values(["Language", "Split"], 
                            ascending=[True, False], 
                            key=lambda x: x.map({"TOTAL": 0, "TRAIN": 1, "TEST": 2, "VAL": 3}), 
                            inplace=True)


    # 保存到CSV（包含完整统计信息）
    final_report.to_csv(
        "msgkeytword.csv", 
        index=False,
        encoding="utf-8-sig",  # 支持Excel中文显示
        columns=[
            "Language", "Split", "Total",
            "Strong_Total", "Strong_Label1", "Strong_Label0",
            "Medium_Total", "Medium_Label1", "Medium_Label0",
            "Both_Total", "Both_Label1", "Both_Label0",
            "Unmatched_Total", "Unmatched_Label1", "Unmatched_Label0"
        ]
    )
    # 打印美观的表格
    print(final_report.to_markdown(index=False, floatfmt=".0f"))