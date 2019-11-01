from responder import *

class Ptna:
    """ ピティナの本体クラス
    """
    def __init__(self, name):
        """ Ptnaオブジェクトの名前をnameに格納
            応答オブジェクトをランダムに生成してresponderに格納

            @param name Ptnaオブジェクトの名前
        """
        self.name = name
        # RandomResponderを生成
        self.res_random = RandomResponder('Random')
        # RepeatResponderを生成
        self.res_what = RepeatResponder('Repeat')
        # responderの初期値をRepeatResponderにする
        self.responder = self.res_what

    def dialogue(self, input):
        """ 応答オブジェクトのresponse()を呼び出して
            応答文字列を取得する

            @param input ユーザーによって入力された文字列
            戻り値 応答文字列
        """
        # 0か1をランダムに生成
        x = random.randint(0, 1)
        # 0ならRandomResponderオブジェクトにする
        if x==0:
            self.responder = self.res_random
        # 0以外ならWhatResponderオブジェクトにする
        else:
            self.responder = self.res_what
        return self.responder.response(input)

    def get_responder_name(self):
        """ 応答オブジェクトの名前を返す
        """
        return self.responder.name

    def get_name(self):
        """ Ptnaオブジェクトの名前を返す
        """
        return self.name
