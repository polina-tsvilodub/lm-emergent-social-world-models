from sentence_transformers import SentenceTransformer
import numpy as np
import os
import nltk
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns

model = SentenceTransformer("all-MiniLM-L6-v2")

original_suites = ["exp2_fb_original_fb_2a", "exp2_d_original_d", "exp2_fp_original_fp_2a", "exp2_h_original",
                   "exp1_mi_original", "exp2_nh_original", "deceptive_dec", "deceptive_iro", "deceptive_lit",
                   "deceptive_mls", "strategic_games_eb", "strategic_games_eo", "moral_intent_tom", "moral_intent_control_new"]
synthetic_suites = ["exp2_fb_synthetic_fb_2a", "exp2_d_synthetic_d_2a", "exp2_fp_synthetic_fp_2a", "exp2_h_synthetic_2a",
                    "mechanical_inference_synthetic", "exp2_nh_synthetic_2a", "deceptive_dec_synthetic", "deceptive_iro_synthetic",
                    "deceptive_lit_synthetic", "deceptive_mls_synthetic", "strategic_games_synthetic_eb",
                    "strategic_games_eo_synthetic_paired", "moral_intent_tom_synthetic", "moral_intent_control_synthetic"
                    ]
conditions = ["FalseBelief", "Desire", "FalsePhotograph", "HumanDescr", "MechInf", "NonhumanDescr", "Deceptive",
              "Ironic", "Literal", "Meaningless", "GameBelief", "GameOutcome", "MoralIntent", "DecOutcome"]

fig, axs = plt.subplots(nrows=7, ncols=2)

for i in range(len(original_suites)):
    suite_original = original_suites[i]
    suite_synthetic = synthetic_suites[i]
    condition = conditions[i]

    items_1 = []
    path_original_stimuli = f"../llm_localizer/stimuli/original_stimuli/{suite_original}"
    for fn in os.listdir(path_original_stimuli):
        if fn not in ["dec_question.txt", "iro_question.txt", "lit_question.txt", "mls_question.txt"] and not fn.endswith("question.txt") and not fn.endswith("aa_story.txt") and not fn.endswith("ab_story.txt") and not fn.endswith("ba_story.txt") and not fn.endswith("question_new.txt") and not fn.endswith("mic_story.txt"):
            items_1.append(nltk.tokenize.sent_tokenize(open(path_original_stimuli + "/" + fn, "r").read()))

    items_2 = []
    path_synthetic_stimuli = f"../llm_localizer/stimuli/synthetic_stimuli/{suite_synthetic}/"
    for fn in os.listdir(path_synthetic_stimuli):
        if fn not in ["decs_question.txt", "iros_question.txt", "lits_question.txt", "mlss_question.txt"] and not fn.endswith("question.txt") and not fn.endswith("aa_story.txt") and not fn.endswith("ab_story.txt") and not fn.endswith("ba_story.txt") and not fn.endswith("question_new.txt") and not fn.endswith("mic_story.txt"):
            items_2.append(nltk.tokenize.sent_tokenize(open(path_synthetic_stimuli + "/" + fn, "r").read()))

    import random
    random.seed(42)
    random.shuffle(items_2)
    items_2 = items_2[:len(items_1)]

    embeddings_1 = [np.sum(model.encode(item), axis=0) for item in items_1]
    embeddings_2 = [np.sum(model.encode(item), axis=0) for item in items_2]
    embeddings = embeddings_1 + embeddings_2

    pca = PCA(n_components=2)

    embeddings = pca.fit_transform(embeddings)

    x_pos = i%2
    y_pos = i//2
    axs[y_pos,x_pos].scatter([embeddings[i][0] for i in range(len(embeddings_1))], [embeddings[i][1] for i in range(len(embeddings_1))], c=sns.color_palette("Reds")[3], label="original")
    axs[y_pos,x_pos].scatter([embeddings[i+len(embeddings_1)][0] for i in range(len(embeddings_2))], [embeddings[i+len(embeddings_1)][1] for i in range(len(embeddings_2))],
            c=sns.color_palette("Blues")[3], label="synthetic (GPT-5)")
    axs[y_pos,x_pos].set_title(condition)
    axs[y_pos,x_pos].legend()
    axs[y_pos,x_pos].set_xlabel("PC 1")
    axs[y_pos,x_pos].set_ylabel("PC 2")
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.5, hspace=0.5)
plt.show()