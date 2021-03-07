import pandas as pd
from sklearn.model_selection import train_test_split

#Read features
with open("spambase.names") as f:
  lines = f.readlines()

features = []
counter = 0
for line in lines:
  counter += 1
  if len(line) < 10:
    continue
  if "word_freq_" == line[:10] or "char_freq_" == line[:10] or counter > len(lines) - 3:
    line = line.strip()
    line = line.split(" ")
    line = line[0]
    feature = line[:-1]
    features.append(feature)
  else:
    continue

features.append("label")

with open("spambase.data") as f:
  lines = f.readlines()

#Read data
data = []
for line in lines:
  line = line.strip()
  line = line.split(",")
  line = [float(i) for i in line]
  data.append(line)

df = pd.DataFrame(data, columns=features)

X = df.drop(columns="label")
y = df["label"].replace(0, -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)