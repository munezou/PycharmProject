from responder import *
from dictionary import *

class Ptna:
    """ ピティナの本体クラス
    """
    def __init__(self, name):
        """ Ptnaオブジェクトの名前をnameに格納
            応答オブジェクトをランダムに生成してresponderに格納

            @param name Ptnaオブジェクトの名前
        """
        self.name = name
        # Dictionaryを生成
        self.dictionary = Dictionary()
        # RandomResponderを生成
        self.res_random = RandomResponder('Random', self.dictionary)
        # RepeatResponderを生成
        self.res_what = RepeatResponder('Repeat', self.dictionary)
        # PatternResponderを生成
        self.res_pattern = PatternResponder('Pattern', self.dictionary)

    def dialogue(self, input):
        """ 応答オブジェクトのresponse()を呼び出して
            応答文字列を取得する

            @param input ユーザーによって入力された文字列
            戻り値 応答文字列
        """
        # 1から100をランダムに生成
        x = random.randint(0, 100)
        # 60以下ならPatternResponderオブジェクトにする
        if x <= 60:
            self.responder = self.res_pattern
        # 61〜90以下ならRandomResponderオブジェクトにする
        elif 61 <= x <= 90:
            self.responder = self.res_random
        # それ以外はRepeatResponderオブジェクトにする
        else:
            self.responder = self.res_what
        return self.responder.response(input)
