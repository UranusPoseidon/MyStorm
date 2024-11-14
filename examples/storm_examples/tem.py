import pandas as pd

# 加载CSV文件
df = pd.read_csv('D:\\STORM\\storm-main\\examples\\costorm_examples\\corpus.csv')

# 检查是否有NaN值
print(df.isna().sum())

# 如果page_content字段有NaN值，删除这些行或填充
df['page_content'].fillna('', inplace=True)

# 保存清理后的CSV文件
df.to_csv('D:\\STORM\\storm-main\\examples\\costorm_examples\\corpus.csv', index=False)