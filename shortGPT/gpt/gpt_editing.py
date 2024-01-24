from shortGPT.gpt import gpt_utils
import json


def getImageQueryPairs(captions, n=15, maxTime=2):
    # Load prompt template
    chat, _ = gpt_utils.load_local_yaml_prompt('prompt_templates/editing_generate_images.yaml')

    # Replace placeholders in the prompt with original captions and other parameters
    prompt = chat.replace('<<CAPTIONS TIMED>>', f"{captions}").replace("<<NUMBER>>", f"{n}")

    # Get completion from ChatGPT
    res = gpt_utils.gpt3Turbo_completion(chat_prompt=prompt, model="gpt-3.5-turbo-1106")

    # Process the response and extract image queries
    imagesCouples = ('{' + res).replace('{', '').replace('}', '').replace('\n', '').split(',')
    pairs = []
    t0 = 0
    end_audio = captions[-1][0][1]
    for a in imagesCouples:
        try:
            query = a[a.find("'") + 1:a.rfind("'")]
            time = float(a.split(":")[0].replace(' ', ''))
            if (time > t0 and time < end_audio):
                pairs.append((time, query + " image"))
                t0 = time
        except:
            print('problem extracting image queries from ', a)
    for i in range(len(pairs)):
        if (i != len(pairs) - 1):
            end = pairs[i][0] + maxTime if (pairs[i + 1][0] - pairs[i][0]) > maxTime else pairs[i + 1][0]
        else:
            end = pairs[i][0] + maxTime if (end_audio - pairs[i][0]) > maxTime else end_audio
        pairs[i] = ((pairs[i][0], end), pairs[i][1])
    return pairs


def getVideoSearchQueriesTimed(captions_timed):
    # Load prompt template
    chat, system = gpt_utils.load_local_yaml_prompt('prompt_templates/editing_generate_videos.yaml')

    # Combine captions_timed without any preprocessing
    captions_timed_combined = ' '.join([caption for (_, caption) in captions_timed])

    # Replace placeholders in the prompt with original captions and other parameters
    chat = chat.replace("<<TIMED_CAPTIONS>>", f"{captions_timed_combined}")

    # Get completion from ChatGPT until the timing matches the end
    end = captions_timed[-1][0][1]
    out = [[[0, 0], ""]]
    while out[-1][0][1] != end:
        try:
            out = json.loads(gpt_utils.gpt3Turbo_completion(chat_prompt=chat, system=system, temp=1).replace("'", '"'),
                             model="gpt-3.5-turbo-1106")
        except Exception as e:
            print(e)
            print("not the right format")

    return out
