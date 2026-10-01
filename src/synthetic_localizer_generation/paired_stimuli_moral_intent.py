import os
import random

def get_paired_prompt_moral_intent(reference_seeds, reference_stories, reference_questions, source_story):
    references = list(zip(reference_seeds, reference_stories, reference_questions))
    random.shuffle(references)
    ref_seeds = [item[0] for item in references]
    ref_stories = [item[1] for item in references]
    ref_questions = [item[2] for item in references]

    #prompt = (f"For the given story, generate a question analogous to the examples.\n\nExamples:\n\n"
    #          f"{ref_stories[0]}\n\n{ref_questions[0]}\n\n----\n\n"
    #          f"{ref_stories[1]}\n\n{ref_questions[1]}\n\n----\n\n"
    #          f"{ref_stories[2]}\n\n{ref_questions[2]}\n\n----\n\n"
    #          f"{ref_stories[3]}\n\n{ref_questions[3]}\n\n----\n\n"
    #          f"{ref_stories[4]}\n\n{ref_questions[4]}\n\n----\n\n"
    #          f"Story:\n\n{source_story}")
    prompt = (f"Convert the given story into a similar story (accompanied by a corresponding question) where no moral "
              f"decision is involved, in analogy to the given examples.\n\nExamples:\n\n"
              f"Original:\n\n{ref_seeds[0]}\n\nNew story:\n\n{ref_stories[0]}\n\nQuestion:\n\n{ref_questions[0]}\n\n----\n\n"
              f"Original:\n\n{ref_seeds[1]}\n\nNew story:\n\n{ref_stories[1]}\n\nQuestion:\n\n{ref_questions[1]}\n\n----\n\n"
              f"Original:\n\n{ref_seeds[2]}\n\nNew story:\n\n{ref_stories[2]}\n\nQuestion:\n\n{ref_questions[2]}\n\n----\n\n"
              f"Original:\n\n{ref_seeds[3]}\n\nNew story:\n\n{ref_stories[3]}\n\nQuestion:\n\n{ref_questions[3]}\n\n----\n\n"
              f"Original:\n\n{ref_seeds[4]}\n\nNew story:\n\n{ref_stories[4]}\n\nQuestion:\n\n{ref_questions[4]}\n\n----\n\n"
              f"Original story:\n\n{source_story}")

    return prompt

if __name__ == "__main__":
    reference_dir_path = "./stimuli/moral_intent_control/"
    source_dir_path = "./stimuli/moral_intent_tom/"
    ref_tag = "mic"
    src_tag = "mor_bb"

    reference_seeds = [open(f"{source_dir_path}{i}{src_tag}_story.txt","r").read() for i in range(100) if os.path.isfile(f"{source_dir_path}{i}{src_tag}_story.txt")]
    reference_stories = [open(f"{reference_dir_path}{i}{ref_tag}_b_story.txt","r").read() for i in range(100) if os.path.isfile(f"{reference_dir_path}{i}{ref_tag}_b_story.txt")]
    reference_questions = [open(f"{reference_dir_path}{i}{ref_tag}_question.txt","r").read() for i in range(100) if os.path.isfile(f"{reference_dir_path}{i}{ref_tag}_question.txt")]
    source_stories = [open(f"{source_dir_path}{i}{src_tag}s_story.txt","r").read() for i in range(100) if os.path.isfile(f"{source_dir_path}{i}{src_tag}s_story.txt")]
    from openai import OpenAI

    client = OpenAI(
        api_key='YOUR_API_KEY'
    )

    for i in range(len(source_stories)):
        prompt = get_paired_prompt_moral_intent(reference_seeds, reference_stories, reference_questions, source_stories[i])
        messages = [{"role": "user", "content": prompt}]
        chat = client.chat.completions.create(model="gpt-5", messages=messages)
        reply = chat.choices[0].message.content
        items = reply.split("\n\nQuestion:\n\n")
        outstory = open(f"{reference_dir_path}{i}{ref_tag}_bsp_story.txt","w")
        outstory.write(items[0])
        outstory.close()
        outquestion = open(f"{reference_dir_path}{i}{ref_tag}_bsp_question.txt","w")
        outquestion.write(items[1])
        outquestion.close()
