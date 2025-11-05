import pandas as pd
import numpy as np
from skrub import TextEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import balanced_accuracy_score, classification_report

# Load tab-separated file
df = pd.read_csv("test_pairs_expr.txt", sep="\t", header=None, names=["entity_1", "entity_2", "label"])


encoder = TextEncoder()
encoder.fit(pd.concat([df["entity_1"]]))

vec_1 = encoder.transform(df["entity_1"])
vec_2 = encoder.transform(df["entity_2"])

similarities = np.diag(cosine_similarity(vec_1.values, vec_2.values))

df["similarity"] = similarities

# Apply threshold
threshold = 0.7
df["predicted"] = (df["similarity"] < threshold).astype(int)
print(df["predicted"])
print(df["label"])

# Evaluate
print(classification_report(df["label"], df["predicted"],digits=3))
