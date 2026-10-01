import os
import random

def get_paired_prompt(reference_stories, reference_questions, source_story):
    references = list(zip(reference_stories, reference_questions))
    random.shuffle(references)
    ref_stories = [item[0] for item in references]
    ref_questions = [item[1] for item in references]

    prompt = (f"For the given story, generate a question analogous to the examples.\n\nExamples:\n\n"
              f"{ref_stories[0]}\n\n{ref_questions[0]}\n\n----\n\n"
              f"{ref_stories[1]}\n\n{ref_questions[1]}\n\n----\n\n"
              f"{ref_stories[2]}\n\n{ref_questions[2]}\n\n----\n\n"
              f"{ref_stories[3]}\n\n{ref_questions[3]}\n\n----\n\n"
              f"{ref_stories[4]}\n\n{ref_questions[4]}\n\n----\n\n"
              f"Story:\n\n{source_story}")
    print(prompt)
    return prompt

if __name__ == "__main__":
    reference_dir_path = "./stimuli/strategic_games_eo/"
    source_dir_path = "./stimuli/strategic_games_eb/"
    ref_tag = "eo"
    src_tag = "ebs"

    reference_stories = [open(f"{reference_dir_path}{i}{ref_tag}_story.txt","r").read() for i in range(100) if os.path.isfile(f"{reference_dir_path}{i}{ref_tag}_story.txt")]
    reference_questions = [open(f"{reference_dir_path}{i}{ref_tag}_question.txt","r").read() for i in range(100) if os.path.isfile(f"{reference_dir_path}{i}{ref_tag}_question.txt")]
    source_stories = [open(f"{source_dir_path}{i}{src_tag}_story.txt","r").read() for i in range(100) if os.path.isfile(f"{source_dir_path}{i}{src_tag}_story.txt")]

    from openai import OpenAI

    client = OpenAI(
        api_key='YOUR_API_KEY'
    )

    for i in range(len(source_stories)):
        prompt = get_paired_prompt(reference_stories, reference_questions, source_stories[i])
        messages = [{"role": "user", "content": prompt}]
        chat = client.chat.completions.create(model="gpt-5", messages=messages)
        reply = chat.choices[0].message.content
        outfile = open(f"{reference_dir_path}{i}{ref_tag}sp_question.txt","w")
        outfile.write(reply)
        outfile.close()