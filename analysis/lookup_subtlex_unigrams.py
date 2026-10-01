import pandas as pd
import numpy as np

# Read frequency data.
freq = pd.read_csv("SUBTLEXus74286wordstextversion.txt", delimiter="\t").set_index("Word")
print(freq.head())

# Read experimental stimuli.
df = pd.read_csv("../stimuli/stimuli.csv")
for i, row in df.iterrows():
    words = row.sentence.split() #[w.lower() for w in row.sentence.split()]
    log_freqs = {}
    for word in words:
        if word not in freq.index:
            word = word.lower() # try the lower-case version
            if word not in freq.index:
                # try capitalizing
                word = word.capitalize()
                if word not in freq.index:
                    print("Skipping word:", word)
                    continue
        word_freq_data = freq.loc[word]
        log_freqs[word] = word_freq_data.Lg10WF
    df.loc[i, "mean_log_freq"] = np.mean(list(log_freqs.values()))
    df.loc[i, "log_freqs"] = str(log_freqs)

df.to_csv("stimuli_log_freq.csv", index=False)