from janome.tokenizer import Tokenizer # janome.tokenizerをインポート
import re

def analyze(text):
    """ 形態素解析を行う

        @param text  解析対象の文章
        戻り値       形態素と品詞のペアを格納した多重リスト
    """    
    t = Tokenizer()                 # Tokenizerオブジェクトを生成
    tokens = t.tokenize(text)       # 形態素解析を実行
    result = []                     # 解析結果の形態素と品詞を格納するリスト
    
    # リストからTokenオブジェクトを1つずつ取り出す
    for token in tokens:
        result.append(
            [token.surface,         # 形態素を取得
             token.part_of_speech]) # 品詞情報を取得
    return(result)

def keyword_check(part):
    """ 品詞が名詞であるか調べる

        @param part  形態素解析の品詞の部分
        戻り値       名詞であればTrue、そうでなければFalse
    """
    return re.match('名詞,(一般|固有名詞|サ変接続|形容動詞語幹)', part)
