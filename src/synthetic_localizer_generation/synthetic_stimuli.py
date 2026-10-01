import random

def get_prompt(stories, questions, no_items="twenty"):
    inputs = list(zip(stories, questions))
    random.shuffle(inputs)
    stories = [item[0] for item in inputs]
    questions = [item[1] for item in inputs]

    prompt = (f"Please generate {no_items} more stories and questions following the structure of "
                      f"the given examples:\n\n----\nStory:\n{stories[0]}\n\nQuestion:\n{questions[0]}\n\n----\n"
                      f"Story:\n{stories[1]}\n\nQuestion:\n{questions[1]}\n\n----\n"
                      f"Story:\n{stories[2]}\n\nQuestion:\n{questions[2]}\n\n----\n"
                      f"Story:\n{stories[3]}\n\nQuestion:\n{questions[3]}\n\n----\n"
                      f"Story:\n{stories[4]}\n\nQuestion:\n{questions[4]}\n\n----\n")
    return prompt

if __name__ == "__main__":
    dir_path = "./PATH_TO_REFERENCE_STIMULI"
    output_path = "./OUTPUT_LOCATION"
    tag = "lb_fb"
    import os

    reference_stories = [open(f"{dir_path}{i}{tag}_story.txt", "r").read() for i in range(100) if
                       os.path.isfile(f"{dir_path}{i}{tag}_story.txt")]
    reference_questions = [open(f"{dir_path}{i}{tag}_question.txt", "r").read() for i in range(100) if
                         os.path.isfile(f"{dir_path}{i}{tag}_question.txt")]

    from openai import OpenAI

    client = OpenAI(
        api_key='YOUR_API_KEY'
    )

    # generate 5×20 synthetic stimuli
    for i in range(5):
        prompt = get_prompt(reference_stories, reference_questions)
        messages = [{"role": "user", "content": prompt}]
        chat = client.chat.completions.create(model="gpt-5", messages=messages)
        reply = chat.choices[0].message.content
        outfile = open(f"{output_path}synthetic_{tag}_{i}.txt", "w")
        outfile.write(reply)
        outfile.close()

    # split these 5 files into individual items
    all_non_tom_stimuli = []
    for i in range(5):
        infile = open(f"{output_path}synthetic_{tag}_{i}.txt", "r")
        examples = infile.read().split("\n\n----\nStory:\n")
        # examples = infile.read().split("\n\nStory:\n")
        for e in examples:
            all_non_tom_stimuli.append(e.split("\n\nQuestion:\n"))
    for i in range(len(all_non_tom_stimuli)):
        outstory = open(f"{output_path}/{i}{tag}_story.txt", "w")
        outstory.write(all_non_tom_stimuli[i][0])
        outstory.close()
        outquestion = open(f"{output_path}/{i}{tag}_question.txt", "w")
        outquestion.write(all_non_tom_stimuli[i][1])
        outquestion.close()
