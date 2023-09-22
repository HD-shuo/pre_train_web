import pandas as pd

lines = []
with open('tokenizer/data/for_test.txt', 'r') as f:
    for line in f:
        lines.append(line.strip())

df = pd.DataFrame({'content': lines})
df.to_parquet('/home/daixingshuo/pre_train_web/data/for_test.parquet')