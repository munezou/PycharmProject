from janome.tokenizer import Tokenizer
import re

def analyze(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    result = []
    for token in tokens:
        result.append(
            [token.surface,
             token.part_of_speech])
    return(result)

def keyword_check(part):
    return re.match('名詞,(一般|固有名詞|サ変接続|形容動詞語幹)', part)
