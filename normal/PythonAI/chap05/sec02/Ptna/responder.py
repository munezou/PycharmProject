import random

class Responder:
    """ 応答クラスのスーパークラス
    """
    def __init__(self, name):
        """ Responderオブジェクトの名前をnameに格納

            @param name Responderオブジェクトの名前
        """
        self.name = name

    def response(self, input):
        """ オーバーライドを前提としたresponse()メソッド

            @param  input 入力された文字列
            戻り値  空の文字列
        """
        return ''

    def get_name(self):
        """ 応答オブジェクトの名前を返す
        """
        return self.name

class RepeatResponder(Responder):
    """ オウム返しのための行うサブクラス
    """
    def response(self, input):
        """ 応答文字列を作って返す

            @param  input 入力された文字列
        """
        return '{}ってなに？'.format(input)

class RandomResponder(Responder):
    """ ランダムな応答のための行うサブクラス
    """
    def __init__(self, name):
        """ ①Responderオブジェクトの名前を引数にして
            スーパークラスの__init__()を呼び出す
            ②ランダム辞書をリストとして読み込んでresponsesに格納

            @param name Responderオブジェクトの名前
        """
        super().__init__(name)

        # ランダム辞書のデータをリストとして保持するインスタンス変数
        self.responses = []
        # ランダム辞書をオープン
        rfile = open('dics/random.txt', 'r', encoding = 'utf_8')
        # 各行を要素とするリストを取得
        r_lines = rfile.readlines()
        rfile.close()
        # 末尾の改行と空白文字を取り除いて
        # インスタンス変数（リスト）に格納
        for line in r_lines:
            str = line.rstrip('\n')
            if (str!=''):
                self.responses.append(str)
    
    def response(self, input):
        """ 応答文字列を作って返す

            @param  input 入力された文字列
            戻り値  リストからランダムに抽出した文字列
        """
        return (random.choice(self.responses))
