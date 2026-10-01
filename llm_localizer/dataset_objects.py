import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import Dataset
import csv
import random
import os
from utils import read_story, read_question

class MDLocDataset(Dataset):
    def __init__(self):  
        num_examples = 100

        self.positive = []
        np.random.seed(42)
        for idx in range(num_examples):
            num_1 = np.random.randint(100, 200)
            num_2 = np.random.randint(100, 200)
            add_or_subtract = np.random.choice(["+", "-"])
            if add_or_subtract == "+":
                question = f"Solve {num_1} + {num_2}?"
                answer = num_1 + num_2
            else:
                question = f"Solve {num_1} - {num_2}?"
                answer = num_1 - num_2
            self.positive.append(f"Question: {question}\nAnswer: {answer}")

        self.negative = []
        np.random.seed(42)
        for idx in range(num_examples):
            num_1 = np.random.randint(1, 20)
            num_2 = np.random.randint(1, 20)
            add_or_subtract = np.random.choice(["+", "-"])
            if add_or_subtract == "+":
                question = f"Solve {num_1} + {num_2}?"
                answer = num_1 + num_2
            else:
                question = f"Solve {num_1} - {num_2}?"
                answer = num_1 - num_2
            self.negative.append(f"Question: {question}\nAnswer: {answer}")

    def __getitem__(self, idx):
        return self.positive[idx].strip(), self.negative[idx].strip()
        
    def __len__(self):
        return len(self.positive) 

class LangLocDataset(Dataset):
    def __init__(self):
        dirpath = "llm_localizer/stimuli/language"
        paths = glob(f"{dirpath}/*.csv")
        vocab = set()

        data = pd.read_csv(paths[0])
        for path in paths[1:]:
            run_data = pd.read_csv(path)
            data = pd.concat([data, run_data])

        data["sent"] = data["stim2"].apply(str.lower)

        vocab.update(data["stim2"].apply(str.lower).tolist())
        for stimuli_idx in range(3, 14):
            data["sent"] += " " + data[f"stim{stimuli_idx}"].apply(str.lower)
            vocab.update(data[f"stim{stimuli_idx}"].apply(str.lower).tolist())

        self.vocab = sorted(list(vocab))
        self.w2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2w = {i: w for i, w in enumerate(self.vocab)}

        self.positive = data[data["stim14"]=="S"]["sent"]
        self.negative = data[data["stim14"]=="N"]["sent"]

    def __getitem__(self, idx):
        return self.positive.iloc[idx].strip(), self.negative.iloc[idx].strip()
        
    def __len__(self):
        return len(self.positive)

class TOMLocDataset(Dataset):
    def __init__(self, dataset_name="tom", without_answers=True):
        if dataset_name == "tom":
            dirpath = "llm_localizer/stimuli/tom/tomloc"
        elif dataset_name == "tom_synthetic":
            dirpath = "llm_localizer/stimuli/tom_synthetic"
        elif dataset_name == "tom_fb_synthetic":
            dirpath = "llm_localizer/synthetic_stimuli/exp2_fb_synthetic_fb_2a"
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")
        
        instruction = "In this experiment, you will read a story, and judge whether a statement about the story is True or False."
        if without_answers:
            context_template = "{instruction}\nStory: {story}\nQuestion: {question}\nAnswer:"
        else:
            context_template = "{instruction}\nStory: {story}\nQuestion: {question}\nAnswer: {answer}"
        belief_stories = [read_story(f"{dirpath}/{idx}b_story.txt") for idx in range(1, 11)]
        photograph_stories = [read_story(f"{dirpath}/{idx}p_story.txt") for idx in range(1, 11)]

        belief_question = [read_question(f"{dirpath}/{idx}b_question.txt") for idx in range(1, 11)]
        photograph_question = [read_question(f"{dirpath}/{idx}p_question.txt") for idx in range(1, 11)]
        np.random.seed(42)
        if without_answers:
            self.positive = [context_template.format(instruction=instruction, story=story, question=question) for story, question in zip(belief_stories, belief_question)]
            self.negative = [context_template.format(instruction=instruction, story=story, question=question) for story, question in zip(photograph_stories, photograph_question)]

        else:
            self.positive = [context_template.format(instruction=instruction, story=story, question=question, answer=np.random.choice(["True", "False"])) for story, question in zip(belief_stories, belief_question)]
            self.negative = [context_template.format(instruction=instruction, story=story, question=question, answer=np.random.choice(["True", "False"])) for story, question in zip(photograph_stories, photograph_question)]

    def __getitem__(self, idx):
        return self.positive[idx].strip(), self.negative[idx].strip()
    
    def __len__(self):
        return len(self.positive)

def rephrase_question(text):
    as_list = text.split("|")
    if len(as_list) != 4:
        return text
    if len(as_list[3].strip()) > 0:
        as_list[1] = as_list[1] + as_list[3].replace(".","").replace("\n"," ").replace("  "," ")
        as_list[2] = as_list[2] + as_list[3].replace(".","").replace("\n"," ").replace("  "," ")
    full_text = "|".join([as_list[0],as_list[1],as_list[2]]).replace("\n"," ").replace("\t"," ") 
    while "  " in full_text:
        full_text = full_text.replace("  "," ")
    return full_text+"|"

class SyntheticActivationsDataset(Dataset):
    """
    Dataset for runninf the modek
    """
    def __init__(self, dataset_name="tom_fb_synthetic", without_answers=True):
        if dataset_name == "tom_fb_synthetic":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/exp2_fb_synthetic_fb_2a"
        elif dataset_name ==  "tom_fp_synthetic":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/exp2_fp_synthetic_fp_2a"
        elif dataset_name == "h_synthetic":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/exp2_h_synthetic_2a"
        elif dataset_name == "nh_synthetic":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/exp2_nh_synthetic_2a"
        elif dataset_name == "tom_d_synthetic":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/exp2_d_synthetic_d_2a"
        elif dataset_name == "mechanical_inference":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/mechanical_inference_synthetic"
        elif dataset_name == "strategic_games_synthetic_eb":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/strategic_games_synthetic_eb"
        elif dataset_name == "strategic_games_eo_synthetic_paired":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/strategic_games_eo_synthetic_paired"
        elif dataset_name == "moral_intent_tom_synthetic":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/moral_intent_tom_synthetic"
        elif dataset_name == "moral_intent_control_synthetic":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/moral_intent_control_synthetic"
        elif dataset_name == "deceptive_mls_synthetic":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/deceptive_mls_synthetic"
        elif dataset_name == "deceptive_lit_synthetic":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/deceptive_lit_synthetic"
        elif dataset_name == "deceptive_iro_synthetic":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/deceptive_iro_synthetic"
        elif dataset_name == "deceptive_dec_synthetic":
            dirpath = "llm_localizer/stimuli/synthetic_stimuli/deceptive_dec_synthetic"

        elif dataset_name == "tom_fb_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/exp2_fb_original_fb_2a"
        elif dataset_name == "tom_fp_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/exp2_fp_original_fp_2a"
        elif dataset_name == "h_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/exp2_h_original_2a"
        elif dataset_name == "nh_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/exp2_nh_original_2a"
        elif dataset_name == "mi_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/exp1_mi_original"
        elif dataset_name == "moral_intent_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/moral_intent_tom"
        elif dataset_name == "moral_intent_control_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/moral_intent_control_new"
        elif dataset_name == "strategic_games_eb_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/strategic_games_eb"
        elif dataset_name == "strategic_games_eo_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/strategic_games_eo"
        elif dataset_name == "deceptive_dec_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/deceptive_dec"
        elif dataset_name == "deceptive_iro_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/deceptive_iro"
        elif dataset_name == "deceptive_lit_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/deceptive_lit"
        elif dataset_name == "deceptive_mls_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/deceptive_mls"
        elif dataset_name == "tom_d_original":
            dirpath = "llm_localizer/stimuli/original_stimuli/exp2_d_original_d"
            dirpath_question = "llm_localizer/stimuli/original_stimuli/exp2_d_original_d_2a"
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")
        
        np.random.seed(42)

        instruction = "In this experiment, you will read a story, a question or a statement and select the best answer among the provided options.\n"
        context_template = "{instruction}Story: {story}\nStatement / question: {question}\nOptions:\n{options}\n"
        
        ultimatum_game_instructions = instruction + "The story is about the Ultimatum Game. There are two players who divide a sum of money in this game. The first player, the proposer, proposes a division of the sum with the second player, the responder. The responder can either accept the proposed division or reject it. If the responder accepts, the money is split according to the proposal; if the responder rejects, neither player receives anything.\n"
        trust_game_instructions = instruction + "The story is about the Trust Game. There are two players who play an investment game. Both players are given some quantity of money. The first player is told that they must send some amount of their money to an anonymous second player, though the amount sent may be zero. The first player is also informed that whatever they send will be tripled by the experimenter and given to the second player. The second player is then told to also give some amount of the now-tripled money back to the first player, even if that amount is zero.\n"
            

        stimuli_files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
        if dataset_name == "mechanical_inference":
            story_numbers = list(set([f.split("_")[0].replace("mis", "") for f in stimuli_files]))
            story_file_template = "{nr}mis_story.txt"
            question_file_template = "{nr}mis_question.txt"
        elif dataset_name == "strategic_games_synthetic_eb":
            story_numbers = list(set([f.split("_")[0].replace("ebs", "") for f in stimuli_files]))
            story_file_template = "{nr}ebs_story.txt"
            question_file_template = "{nr}ebs_question.txt"
        elif dataset_name == "strategic_games_eo_synthetic_paired":
            story_numbers = list(set([f.split("_")[0].replace("ebs", "").replace("eosp", "") for f in stimuli_files]))
            story_file_template = "{nr}ebs_story.txt"
            question_file_template = "{nr}eosp_question.txt"
        elif dataset_name == "moral_intent_tom_synthetic":
            story_numbers = list(set([f.split("_")[0].replace("mor", "") for f in stimuli_files]))
            story_file_template = "{nr}mor_bbs_story.txt"
            question_file_template = "{nr}mor_bbs_question.txt"
        elif dataset_name == "moral_intent_control_synthetic":
            story_numbers = list(set([f.split("_")[0].replace("mic", "") for f in stimuli_files]))
            story_file_template = "{nr}mic_bsp_story.txt"
            question_file_template = "{nr}mic_bsp_question.txt"
        elif dataset_name == "deceptive_dec_synthetic":
            story_numbers = list(set([f.split("_")[0].replace("decs", "") for f in stimuli_files if len(f.split("_")[0].replace("decs", "")) > 0]))
            story_file_template = "{nr}decs_story.txt"
            question_file_template = "decs_question.txt"
        elif dataset_name == "deceptive_iro_synthetic":
            story_numbers = list(set([f.split("_")[0].replace("iros", "") for f in stimuli_files if len(f.split("_")[0].replace("iros", "")) > 0]))
            story_file_template = "{nr}iros_story.txt"
            question_file_template = "iros_question.txt"
        elif dataset_name == "deceptive_lit_synthetic":
            story_numbers = list(set([f.split("_")[0].replace("lits", "") for f in stimuli_files if len(f.split("_")[0].replace("lits", "")) > 0]))
            story_file_template = "{nr}lits_story.txt"
            question_file_template = "lits_question.txt"
        elif dataset_name == "deceptive_mls_synthetic":
            story_numbers = list(set([f.split("_")[0].replace("mlss", "") for f in stimuli_files if len(f.split("_")[0].replace("mlss", "")) > 0]))
            story_file_template = "{nr}mlss_story.txt"
            question_file_template = "mlss_question.txt"


        elif dataset_name == "tom_fb_original":
            story_numbers = list(set([f.split("_")[0].replace("bb2", "") for f in stimuli_files]))
            story_file_template = "{nr}bb2_story.txt"
            question_file_template = "{nr}bb2_question.txt"
        elif dataset_name == "tom_fp_original":
            story_numbers = list(set([f.split("_")[0].replace("pp2", "") for f in stimuli_files]))
            story_file_template = "{nr}pp2_story.txt"
            question_file_template = "{nr}pp2_question.txt"
        elif dataset_name == "mi_original":
            # story_numbers = list(set([f.split("_")[0].replace("mi", "").replace(".txt", "") for f in stimuli_files]))
            story_numbers = ["10", "11", "12", "2", "3", "4", "6", "7", "8"]
            story_file_template = "{nr}mi.txt"
            question_file_template = "{nr}mi_question_new.txt"
        elif dataset_name == "moral_intent_original":
            story_numbers = list(set([f.split("_")[0].replace("mor", "") for f in stimuli_files]))
            story_file_template = "{nr}mor_bb_story.txt"
            question_file_template = "{nr}mor_question.txt"
        elif dataset_name == "moral_intent_control_original":
            story_numbers = list(set([f.split("_")[0].replace("mic", "") for f in stimuli_files]))
            story_file_template = "{nr}mic_b_story.txt"
            question_file_template = "{nr}mic_question.txt"
        elif dataset_name == "strategic_games_eb_original":
            story_numbers = list(set([f.split("_")[0].replace("eb", "") for f in stimuli_files]))
            story_file_template = "{nr}eb_story.txt"
            question_file_template = "{nr}eb_question.txt"
        elif dataset_name == "strategic_games_eo_original":
            story_numbers = list(set([f.split("_")[0].replace("eo", "") for f in stimuli_files]))
            story_file_template = "{nr}eo_story.txt"
            question_file_template = "{nr}eo_question.txt"
        elif dataset_name == "deceptive_dec_original":
            story_numbers = list(set([f.split("_")[0].replace("dec", "") for f in stimuli_files if len(f.split("_")[0].replace("dec", "")) > 0]))
            print("deceptive_dec_original story numbers ", story_numbers)
            story_file_template = "{nr}dec_story.txt"
            question_file_template = "dec_question.txt"
        elif dataset_name == "deceptive_iro_original":
            story_numbers = list(set([f.split("_")[0].replace("iro", "") for f in stimuli_files if len(f.split("_")[0].replace("iro", "")) > 0]))
            story_file_template = "{nr}iro_story.txt"
            question_file_template = "iro_question.txt"
        elif dataset_name == "deceptive_lit_original":
            story_numbers = list(set([f.split("_")[0].replace("lit", "") for f in stimuli_files if len(f.split("_")[0].replace("lit", "")) > 0]))
            story_file_template = "{nr}lit_story.txt"
            question_file_template = "lit_question.txt"
        elif dataset_name == "deceptive_mls_original":
            story_numbers = list(set([f.split("_")[0].replace("mls", "") for f in stimuli_files if len(f.split("_")[0].replace("mls", "")) > 0]))
            story_file_template = "{nr}mls_story.txt"
            question_file_template = "mls_question.txt"

        elif dataset_name == "nh_original":
            dirpath_story = "llm_localizer/stimuli/original_stimuli/exp2_nh_original"
            dirpath_question = "llm_localizer/stimuli/original_stimuli/exp2_nh_original_2a"
            story_stimuli_files = [f for f in os.listdir(dirpath_story) if os.path.isfile(os.path.join(dirpath_story, f))]
            question_stimuli_files = [f for f in os.listdir(dirpath_question) if os.path.isfile(os.path.join(dirpath_question, f))]
            story_numbers = list(set([f.split("_")[0].replace("nh", "") for f in story_stimuli_files]))
            question_numbers = list(set([f.split("_")[0].replace("nh", "") for f in question_stimuli_files]))
            story_file_template = "{nr}nh_story.txt"
            question_file_template = "{nr}nh_question.txt"
        elif dataset_name == "h_original":
            dirpath_story = "llm_localizer/stimuli/original_stimuli/exp2_h_original"
            dirpath_question = "llm_localizer/stimuli/original_stimuli/exp2_h_original_2a"
            story_stimuli_files = [f for f in os.listdir(dirpath_story) if os.path.isfile(os.path.join(dirpath_story, f))]
            question_stimuli_files = [f for f in os.listdir(dirpath_question) if os.path.isfile(os.path.join(dirpath_question, f))]
            story_numbers = list(set([f.split("_")[0].replace("h", "") for f in story_stimuli_files]))
            question_numbers = list(set([f.split("_")[0].replace("h", "") for f in question_stimuli_files]))
            story_file_template = "{nr}h_story.txt"
            question_file_template = "{nr}h_question.txt"
        elif dataset_name == "tom_d_original":
            dirpath_story = "llm_localizer/stimuli/original_stimuli/exp2_d_original_d"
            dirpath_question = "llm_localizer/stimuli/original_stimuli/exp2_d_original_d_2a"
            story_stimuli_files = [f for f in os.listdir(dirpath_story) if os.path.isfile(os.path.join(dirpath_story, f)) if "dd" in f]
            question_stimuli_files = [f for f in os.listdir(dirpath_question) if os.path.isfile(os.path.join(dirpath_question, f)) if "dd" in f]
            story_numbers = list(set([f.split("_")[0].replace("dd", "") for f in story_stimuli_files]))
            question_numbers = list(set([f.split("_")[0].replace("dd2", "") for f in question_stimuli_files]))
            story_file_template = "{nr}dd_story.txt"
            question_file_template = "{nr}dd2_question.txt"

        else:
            story_numbers = list(set([f.split("_")[0].replace("2s", "") for f in stimuli_files]))
            story_file_template = "{nr}2s_story.txt"
            question_file_template = "{nr}2s_question.txt"

        if (dataset_name == "tom_d_original") or (dataset_name == "nh_original") or (dataset_name == "h_original"):
            stories = [read_story(os.path.join(dirpath_story, story_file_template.format(nr=nr))) for nr in story_numbers]
            questions = [read_story(os.path.join(dirpath_question, question_file_template.format(nr=nr))) for nr in question_numbers]
        elif (dataset_name == "deceptive_dec_original") or (dataset_name == "deceptive_iro_original") or (dataset_name == "deceptive_lit_original") or (dataset_name == "deceptive_mls_original"): 
            print("deceptive_dec_original story file template ", story_file_template, question_file_template)
            stories = [read_story(os.path.join(dirpath, story_file_template.format(nr=nr))) for nr in story_numbers]   
            questions = [read_story(os.path.join(dirpath, question_file_template)) for nr in story_numbers ]
        elif (dataset_name == "deceptive_dec_synthetic") or (dataset_name == "deceptive_iro_synthetic") or (dataset_name == "deceptive_lit_synthetic") or (dataset_name == "deceptive_mls_synthetic"): 
            stories = [read_story(os.path.join(dirpath, story_file_template.format(nr=nr))) for nr in story_numbers]   
            questions = [read_story(os.path.join(dirpath, question_file_template)) for nr in story_numbers ]
        else:
            # TODO: when we use moral controls, only use _b ones
            stories = [read_story(os.path.join(dirpath, story_file_template.format(nr=nr))) for nr in story_numbers]
            questions = [read_story(os.path.join(dirpath, question_file_template.format(nr=nr))) for nr in story_numbers ]

        clean_questions_answers = [rephrase_question(q) for q in questions]
        clean_questions = [q.split("|")[0].strip() for q in clean_questions_answers]
        clean_answers = [[a.replace("?", "") for a in q.split("|")[1:] if len(a) > 0] for q in clean_questions_answers]
        shuffled_answers = [np.random.choice(opt, replace=False, size=len(opt)) for opt in clean_answers]
        if "deceptive" in dataset_name:
            formatted_answers = ["\n".join([f"- {o}" for o in opt]) + "?\n" for opt in shuffled_answers]
        else:
            formatted_answers = ["".join([f"- {o}\n" for o in opt]) for opt in shuffled_answers]
        print("clean questions and answers post processing ", clean_questions, clean_answers, formatted_answers)
        ##### TODO: #####
        # add task specific instruction where needed
        if dataset_name == "strategic_games_synthetic_eb":
            instructions_strategic_games_synthetic_eb = [trust_game_instructions if "Trust Game" in story else ultimatum_game_instructions for story in stories]
            self.user_context = [context_template.format(instruction=instruction, story=story, question=question, options=option) for instruction, story, question, option in zip(instructions_strategic_games_synthetic_eb, stories, clean_questions, formatted_answers)]

        elif dataset_name == "strategic_games_synthetic_eo":
            instructions_strategic_games_synthetic_eo = [trust_game_instructions if "Trust Game" in story else ultimatum_game_instructions for story in stories]
            self.user_context = [context_template.format(instruction=instruction, story=story, question=question, options=option) for instruction, story, question, option in zip(instructions_strategic_games_synthetic_eo, stories, clean_questions, formatted_answers)]
        else:
            self.user_context = [context_template.format(instruction=instruction, story=story, question=question, options=option) for story, question, option in zip(stories, clean_questions, formatted_answers)]

        if without_answers:
            # TODO: integrate, and just concat for base models
            if "deceptive" in dataset_name:
                self.assistant_context = [f"Answer: It was" for question in clean_questions] 
            else:    
                self.assistant_context = [f"Answer: {question}" for question in clean_questions] 

        else:
            if "deceptive" in dataset_name:
                self.assistant_context = [f"Answer: It was {random.choice(option)}." for question, option in zip(clean_questions, shuffled_answers)] 
            else:
                self.assistant_context = [f"Answer: {question} {random.choice(option)}." for question, option in zip(clean_questions, shuffled_answers)] 

    def __getitem__(self, idx):
        return self.user_context[idx].strip(), self.assistant_context[idx].strip()

    def __len__(self):
        return len(self.user_context)
    
class TomiFODataset(Dataset):
    def __init__(self):
        dirpath = "behavioral_eval_stimuli/log_p_curated"
        tom = list(csv.reader(open(f"{dirpath}/tomi_tb.csv")))[1:] + list(
            csv.reader(open(f"{dirpath}/tomi_fb.csv")))[1:]
        non_tom = list(csv.reader(open(f"{dirpath}/tomi_reality.csv")))[1:] + list(
            csv.reader(open(f"{dirpath}/tomi_memory.csv")))[1:]
        np.random.seed(42)
        random.seed(42)
        random.shuffle(tom)
        random.shuffle(non_tom)
        number_of_items = min(len(tom), len(non_tom))
        self.positive = [
            f"Read the story and answer the question.\nStory: {item[0]}\nQuestion: {item[1]}\nAnswer: {np.random.choice([item[2], item[3]])}"
            for item in tom][:number_of_items]
        self.negative = [
            f"Read the story and answer the question.\nStory: {item[0]}\nQuestion: {item[1]}\nAnswer: {np.random.choice([item[2], item[3]])}"
            for item in non_tom][:number_of_items]

    def __getitem__(self, idx):
        return self.positive[idx].strip(), self.negative[idx].strip()

    def __len__(self):
        return len(self.positive)

class TomiSODataset(Dataset):
    def __init__(self):
        dirpath = "behavioral_eval_stimuli/log_p_curated"
        tom = list(csv.reader(open(f"{dirpath}/tomi_sotb.csv")))[1:] + list(
            csv.reader(open(f"{dirpath}/tomi_sofb.csv")))[1:]
        non_tom = list(csv.reader(open(f"{dirpath}/tomi_reality.csv")))[1:] + list(
            csv.reader(open(f"{dirpath}/tomi_memory.csv")))[1:]
        np.random.seed(42)
        random.seed(42)
        random.shuffle(tom)
        random.shuffle(non_tom)
        number_of_items = min(len(tom), len(non_tom))
        self.positive = [
            f"Read the story and answer the question.\nStory: {item[0]}\nQuestion: {item[1]}\nAnswer: {np.random.choice([item[2], item[3]])}"
            for item in tom][:number_of_items]
        self.negative = [
            f"Read the story and answer the question.\nStory: {item[0]}\nQuestion: {item[1]}\nAnswer: {np.random.choice([item[2], item[3]])}"
            for item in non_tom][:number_of_items]

    def __getitem__(self, idx):
        return self.positive[idx].strip(), self.negative[idx].strip()

    def __len__(self):
        return len(self.positive)

class FauxpasDataset(Dataset):
    def __init__(self):
        
        dirpath = "behavioral_eval_stimuli/log_p_curated"
        reader = csv.reader(open(f"{dirpath}/fauxpas_eai_q1_and_q4.csv"))
        q1_and_q4 = list(reader)[1:]
        positive_stimuli = [[item[2], item[3], item[4], item[5]] for item in q1_and_q4] + [[item[2], item[6], item[7], item[8]] for item in q1_and_q4]
        np.random.seed(42)
        random.seed(42)
        random.shuffle(positive_stimuli)
        q3 = list(csv.reader(open(f"{dirpath}/fauxpas_eai_q3.csv")))[1:]
        negative_stimuli = [[item[2], item[3], item[4], item[5]] for item in q3]
        number_of_items = min(len(positive_stimuli), len(negative_stimuli))
        self.positive = [f"Read the story and answer the question.\nStory: {item[0]}\nQuestion: {item[1]}\nAnswer: {np.random.choice([item[2], item[3]])}" for item in positive_stimuli][:number_of_items]
        self.negative = [f"Read the story and answer the question.\nStory: {item[0]}\nQuestion: {item[1]}\nAnswer: {np.random.choice([item[2], item[3]])}" for item in negative_stimuli][:number_of_items]

    def __getitem__(self, idx):
        return self.positive[idx].strip(), self.negative[idx].strip()

    def __len__(self):
        return len(self.positive)

class TOMMatchedSyntheticDataset(Dataset):
    def __init__(self, dataset_name="tom_matched_synthetic"):
        if dataset_name == "tom_matched_synthetic":
            dirpath = "llm_localizer/stimuli/tom_matched_synthetic"
        elif dataset_name == "tom_matched":
            dirpath = "llm_localizer/stimuli/tom_matched"
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")
        instruction = "In this experiment, you will read a series of sentences and then answer True/False questions about them."
        context_template = "{instruction}\nStory: {story}\nQuestion: {question}\nAnswer: "
        stories = [read_story(f"{dirpath}/{idx}b_story.txt") for idx in range(200)]

        belief_question = [read_question(f"{dirpath}/{idx}b_question.txt") for idx in range(200)]
        memory_question = [read_question(f"{dirpath}/{idx}bm_question.txt") for idx in range(200)]
        reality_question = [read_question(f"{dirpath}/{idx}br_question.txt") for idx in range(200)]
        nonbelief_question = memory_question + reality_question
        random.seed(42)
        random.shuffle(nonbelief_question)

        self.positive = [context_template.format(instruction=instruction, story=story, question=question) for
                         story, question in zip(stories, belief_question)]
        self.negative = [context_template.format(instruction=instruction, story=story, question=question) for
                         story, question in zip(stories, nonbelief_question)][:len(self.positive)]

    def __getitem__(self, idx):
        return self.positive[idx].strip(), self.negative[idx].strip()

    def __len__(self):
        return len(self.positive)

class PhotographSynthDataset(Dataset):
    def __init__(self):
        dirpath = "llm_localizer/stimuli/photograph_synthetic"
        instruction = "In this experiment, you will read a series of sentences and then answer True/False questions about them."
        context_template = "{instruction}\nStory: {story}\nQuestion: {question}\nAnswer: "
        stories = [read_story(f"{dirpath}/{idx}p_story.txt") for idx in range(200)]

        belief_question = [read_question(f"{dirpath}/{idx}pb_question.txt") for idx in range(200)]
        memory_question = [read_question(f"{dirpath}/{idx}p_question.txt") for idx in range(200)]
        reality_question = [read_question(f"{dirpath}/{idx}pr_question.txt") for idx in range(200)]
        nonbelief_question = memory_question + reality_question
        random.seed(42)
        random.shuffle(nonbelief_question)

        self.positive = [context_template.format(instruction=instruction, story=story, question=question) for
                         story, question in zip(stories, belief_question)]
        self.negative = [context_template.format(instruction=instruction, story=story, question=question) for
                         story, question in zip(stories, nonbelief_question)][:len(self.positive)]

    def __getitem__(self, idx):
        return self.positive[idx].strip(), self.negative[idx].strip()

    def __len__(self):
        return len(self.positive)