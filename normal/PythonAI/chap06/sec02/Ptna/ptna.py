from responder import *
from dictionary import *
from analyzer import *

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
        # Emotionを生成
        self.emotion = Emotion(self.dictionary)
        
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
        # 機嫌値を更新
        self.emotion.update(input)
        # インプット文字列を解析
        parts = analyze(input)
        #print(parts)
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

        # 応答フレーズを生成
        resp = self.responder.response(input, self.emotion.mood)
        # 学習メソッドを呼ぶ
        self.dictionary.study(input, parts)
        # 応答フレーズを返す
        return resp

    def save(self):
        """ Dictionaryのsave()を呼ぶ中継メソッド
        """
        self.dictionary.save()
        

class Emotion:
    """ ピティナの感情モデル
    """
    # 機嫌値の上限／加減と回復値を設定
    MOOD_MIN = -15
    MOOD_MAX = 15
    MOOD_RECOVERY = 0.5

    def __init__(self, dictionary):
        """ Dictionaryオブジェクトをdictionaryに格納
            機嫌値moodを0で初期化

            @param dictionary Dictionaryオブジェクト
        """
        self.dictionary = dictionary
        # 機嫌値を保持するインスタンス変数
        self.mood = 0

    def update(self, input):
        """ ユーザーからの入力をパラメーターinputで受け取り
            パターン辞書にマッチさせて機嫌値を変動させる

            @param input ユーザーからの入力
        """
        # 機嫌を徐々にもとに戻す処理
        if self.mood < 0:
          self.mood += Emotion.MOOD_RECOVERY
        elif self.mood > 0:
          self.mood -= Emotion.MOOD_RECOVERY
        # パターン辞書の各行を繰り返しパターンマッチさせる
        for ptn_item in self.dictionary.pattern:
            # パターンマッチすればadjust_mood()で機嫌値を変動させる
            if ptn_item.match(input):
                self.adjust_mood(ptn_item.modify)
                break

    def adjust_mood(self, val):
        """ 機嫌値を増減させる

            @param val 機嫌変動値
        """
        # 機嫌値moodの値を機嫌変動値によって増減する
        self.mood += int(val)
        # MOOD_MAXとMOOD_MINと比較して、機嫌値が取り得る範囲に収める
        if self.mood > Emotion.MOOD_MAX:
          self.mood = Emotion.MOOD_MAX
        elif self.mood < Emotion.MOOD_MIN:
          self.mood = Emotion.MOOD_MIN
