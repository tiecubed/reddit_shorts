import re
import nltk
from nltk import sent_tokenize

nltk.download('punkt')  # Download the punkt tokenizer data


def getSpeechBlocks(whispered, silence_time=2):
    text_blocks, (st, et, txt) = [], (0, 0, "")
    for i, seg in enumerate(whispered['segments']):
        if seg['start'] - et > silence_time:
            if txt: text_blocks.append([[st, et], txt])
            (st, et, txt) = (seg['start'], seg['end'], seg['text'])
        else:
            et, txt = seg['end'], txt + seg['text']

    if txt: text_blocks.append([[st, et], txt])  # For last text block

    return text_blocks


def cleanWord(word):
    return re.sub(r'[^\w\s\-_"\'\']', '', word)


def interpolateTimeFromDict(word_position, d):
    for key, value in d.items():
        if key[0] <= word_position <= key[1]:
            return value
    return None


def getTimestampMapping(whisper_analysis):
    index = 0
    locationToTimestamp = {}
    for segment in whisper_analysis['segments']:
        for word in segment['words']:
            newIndex = index + len(word['text']) + 1
            locationToTimestamp[(index, newIndex)] = word['end']
            index = newIndex
    return locationToTimestamp


def splitWordsBySize(text, maxCaptionSize=15):
    words = text.split()
    cleaned_words = [cleanWord(word) for word in words]

    captions = []
    current_caption = ""

    for word in cleaned_words:
        if len(current_caption + ' ' + word) <= maxCaptionSize:
            current_caption += ' ' + word
        else:
            captions.append(current_caption.strip())
            current_caption = word

    if current_caption:
        captions.append(current_caption.strip())

    return captions


def getCaptionsWithTime(whisper_analysis, considerPunctuation=False):
    wordLocationToTime = getTimestampMapping(whisper_analysis)
    position = 0
    start_time = 0
    CaptionsPairs = []
    text = whisper_analysis['text']

    if considerPunctuation:
        sentences = sent_tokenize(text)
        words = [cleanWord(word) for sentence in sentences for word in sentence.split()]
    else:
        words = text.split()
        words = [cleanWord(word) for word in words]

    current_caption = ""

    for word in words:
        position += len(word) + 1
        end_time = interpolateTimeFromDict(position, wordLocationToTime)
        if end_time and word:
            CaptionsPairs.append(((start_time, end_time), word))
            start_time = end_time

    return CaptionsPairs
