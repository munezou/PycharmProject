class Ptna:
    """ ピティナの本体クラス
    """
    def __init__(self, name):
        """ Ptnaオブジェクトの名前をnameに格納
            Responderオブジェクトを生成してresponderに格納

            @param name Ptnaオブジェクトの名前
        """
        self.name = name
        self.responder = Responder('What')

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

class Responder:
    """ 応答クラス
    """
    def __init__(self, name):
        """ Responderオブジェクトの名前をnameに格納

            @param name Responderオブジェクトの名前
        """
        self.name = name

    def response(self, input):
        """ 応答文字列を作って返す

            @param input 入力された文字列
        """
        return '{}ってなに？'.format(input)

##    def get_name(self):
##        """ 応答オブジェクトの名前を返す
##        """
##        return self.name

##################################################################
#実行ブロック
##################################################################
def prompt(obj):
    """ ピティナのプロンプトを作る関数
        戻り値 'Ptnaオブジェクト名:応答オブジェクト名 > '
    """
    return obj.get_name() + ':' + obj.get_responder_name() + '> '

print('Ptna System prototype : ptna') # プログラムの情報を表示
ptna = Ptna('ptna')                   # Ptnaオブジェクトを生成

while True:                           # 対話処理開始
    inputs = input(' > ')
    if not inputs:
        print('バイバイ')
        break
    response = ptna.dialogue(inputs)  # 応答文字列を取得
    print(prompt(ptna), response)     # プロンプトと応答文字列をつなげて表示
