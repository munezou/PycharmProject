from responder import *

class Ptna:
    """ ピティナの本体クラス
    """
    def __init__(self, name):
        """ Ptnaオブジェクトの名前をnameに格納
           RandomResponderオブジェクトを生成してresponderに格納

            @param name Ptnaオブジェクトの名前
        """
        self.name = name
        self.responder = RandomResponder('Random')

    def dialogue(self, input):
        """ 応答オブジェクトのresponse()を呼び出して
            応答文字列を取得する

            @param input ユーザーによって入力された文字列
            戻り値 応答文字列
        """
        return self.responder.response(input)

    def get_responder_name(self):
        """ 応答オブジェクトの名前を返す
        """
        return self.responder.name

    def get_name(self):
        """ Ptnaオブジェクトの名前を返す
        """
        return self.name
