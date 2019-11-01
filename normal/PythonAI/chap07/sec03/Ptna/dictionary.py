import random
import re
from analyzer import *
from markov import *


class Dictionary:
    def __init__(self):
        """ 辞書を作成
        """
        self.load_random()
        self.load_pattern()
        self.load_template()
        self.load_markov()


    def load_random(self):
        """ ランダム辞書を作成
        """
        # ランダム辞書を保持するリスト
        self.random = []
        # ランダム辞書ファイルオープン
        rfile = open('dics/random.txt', 'r', encoding = 'utf_8')
        # 各行を要素としてリストに格納
        r_lines = rfile.readlines()
        rfile.close()

        # 末尾の改行と空白文字を取り除いて
        # インスタンス変数（リスト）に格納
        for line in r_lines:
            str = line.rstrip('\n')
            if (str!=''):
                self.random.append(str)

    def load_pattern(self):
        """ パターン辞書を作成
        """
        # パターン辞書を保持するリスト
        self.pattern = []
        # パターン辞書オープン
        pfile = open('dics/pattern.txt', 'r', encoding = 'utf_8')
        # 各行を要素としてリストに格納
        p_lines = pfile.readlines()
        pfile.close()
        # 末尾の改行と空白文字を取り除いて
        # インスタンス変数（リスト）に格納
        self.new_lines = []
        for line in p_lines:
            str = line.rstrip('\n')
            if (str!=''):
                self.new_lines.append(str)

        # 辞書データの各行をタブで切り分ける
        # ptn 正規表現のパターン
        # prs 応答例
        # ParseItemオブジェクトを生成(引数はptn、prs）して
        # インスタンス変数pattern（リスト）に追加
        for line in self.new_lines:
            ptn, prs = line.split('\t')
            self.pattern.append(ParseItem(ptn, prs))

    def load_template(self):
        """ テンプレート辞書を作成
        """
        # テンプレート辞書を保持する辞書
        self.template = {}
        # テンプレート辞書ファイルオープン
        tfile = open('dics/template.txt', 'r', encoding = 'utf_8')
        # 各行を要素としてリストに格納
        t_lines = tfile.readlines()

        tfile.close()

        # 末尾の改行と空白文字を取り除いて
        # インスタンス変数（リスト）に格納
        self.new_t_lines = []
        for line in t_lines:
            str = line.rstrip('\n')
            if (str!=''):
                self.new_t_lines.append(str)

        # テンプレート辞書の各行をタブで切り分ける
        # count    %noun%の出現回数
        # template テンプレート文字列
        # ParseItemオブジェクトを生成(引数はptn、prs）して
        # インスタンス変数pattern（リスト）に追加
        for line in self.new_t_lines:
            # テンプレート行をタブでcount, templateに分割
            count, template = line.split('\t')
            # self.templateのキーにcount(出現回数)が存在しなければ
            # countをキーにして空のリストを要素として追加
            if not count in self.template:
                self.template[count] = []

            # countキーのリストにテンプレート文字列を追加
            self.template[count].append(template)

    def load_markov(self):
        """ マルコフ辞書を作成
        """
        # ログ辞書からマルコフ連鎖で生成した文章を保持するリスト
        self.sentences = []
        # Markovオブジェクトを生成
        markov = Markov()
        # マルコフ連鎖で生成された文章群を取得
        text = markov.make()
        # 各文章の末尾の改行で分割してリストに格納
        self.sentences = text.split('\n')
        # リストから空の要素を取り除く
        if '' in self.sentences:
            self.sentences.remove('')
        
    def study(self, input, parts):
        """ study_random()、study_pattern()、study_template()を呼ぶ

            @param input  ユーザーの発言
            @param parts  形態素解析結果
        """
        # インプット文字列末尾の改行は取り除いておく
        input = input.rstrip('\n')
        # 各学習メソッドを呼び分ける
        # @param input インプット文字列
        # @param parts  インプット文字列の形態素
        self.study_random(input)
        self.study_pattern(input, parts)
        self.study_template(parts)

    def study_random(self, input):
        """ ユーザーの発言を学習する

            @param input  ユーザーの発言
        """
        # 発言がランダム辞書に存在しなければ
        # self.randomの末尾に追加
        if not input in self.random:
            self.random.append(input)

    def study_pattern(self, input, parts):
        """ パターンを学習する

            @param input  ユーザーの発言
            @param input  ユーザーの発言
        """
        for word, part in parts:
            # 名詞であるかをチェック
            if (keyword_check(part)):
                depend = False # ParseItemオブジェクトを保持する変数
                # patternリストのpatternキーを反復処理
                for ptn_item in self.pattern:
                    m = re.search(ptn_item.pattern, word)
                    # インプットされた名詞が既存のパターンとマッチしたら
                    # patternリストからマッチしたParseItemオブジェクトを取得
                    if(re.search(ptn_item.pattern, word)):
                        depend = ptn_item
                        break #マッチしたら止める
                # 既存パターンとマッチしたParseItemオブジェクトから
                # add_phraseを呼ぶ
                if depend:
                    depend.add_phrase(input) # 引数はインプット文字列

                else:
                    # 既存パターンに存在しない名詞であれば
                    # 新規のPatternItemオブジェクトを
                    # patternリストに追加
                    self.pattern.append(ParseItem(word, input))

    def study_template(self, parts):
        """ テンプレートを学習する

        @param parts  形態素解析の結果（リスト）
        """        
        template = ''
        count = 0
        for word, part in parts:
            # 名詞であるかをチェック
            if (keyword_check(part)):
                word = '%noun%'
                count += 1
            template += word

        # self.templateのキーにcount(出現回数)が存在しなければ
        # countをキーにして空のリストを要素として追加
        if count > 0:
            count = str(count)

            if not count in self.template:
                self.template[count] = []
            # countキーのリストにテンプレート文字列を追加
            if not template in self.template[count]:
                self.template[count].append(template)

    def save(self):
        """ self.random、self.pattern、self.templateの内容を
            辞書にファイルに書き込む
        """
        # 各要素の末尾に改行を追加する
        for index, element in enumerate(self.random):
            self.random[index] = element +'\n'
        # ランダム辞書に書き込む
        with open('dics/random.txt', 'w', encoding = 'utf_8') as f:
            f.writelines(self.random)

        pattern = []
        for ptn_item in self.pattern:
           # make_line()呼び出し
           pattern.append(ptn_item.make_line() + '\n')

        # パターン辞書に書き込む
        with open('dics/pattern.txt', 'w', encoding = 'utf_8') as f:
             f.writelines(pattern)


        template = []
        # self.templateのすべてのキーと値のペアを取得して
        # 反復処理を行う
        for key, val in self.template.items():
            # 値のリストをイテレートし
            # 「キー + タブ + リストの個々の要素 + 改行」の1行を作る
            for v in val:
                template.append(key + '\t' + v + '\n')
        # テンプレート辞書に書き込む
        with open('dics/template.txt', 'w', encoding = 'utf_8') as f:
             f.writelines(template)

class ParseItem:
    SEPARATOR = '^((-?\d+)##)?(.*)$'

    def __init__(self, pattern, phrases):
        """ @param pattern  パターン
            @param phrases  応答例
        """
        # 辞書のパターンの部分にSEPARATORをパターンマッチさせる
        m = re.findall(ParseItem.SEPARATOR, pattern)
        # インスタンス変数modifyに0を代入
        self.modify = 0
        # マッチ結果の整数の部分が空でなければ値を代入
        if m[0][1]:
          self.modify =int(m[0][1])
        # インスタンス変数patternにマッチ結果のパターン部分を代入
        self.pattern = m[0][2]
        
        self.phrases = []   # 応答例を保持するインスタンス変数
        self.dic = {}       # インスタンス変数
        # 引数で渡された応答例を'|'で分割し、
        # 個々の要素に対してSEPARATORをパターンマッチさせる
        # self.phrases[ 'need'  : 応答例の整数部分
        #               'phrase': 応答例の文字列部分 ]
        for phrase in phrases.split('|'):
            # 応答例に対してパターンマッチを行う
            m = re.findall(ParseItem.SEPARATOR, phrase)
            # 'need'キーの値を整数部分m[0][1]にする
            # 'phrase'キーの値を応答文字列m[0][2]にする
            self.dic['need'] = 0
            if m[0][1]:
                self.dic['need'] = int(m[0][1])
            self.dic['phrase'] = m[0][2]
            # 作成した辞書をphrasesリストに追加
            self.phrases.append(self.dic.copy())

    def match(self, str):
        """self.pattern(各行ごとの正規表現)を
           インプット文字列にパターンマッチ
        """
        return re.search(self.pattern, str)

    def choice(self, mood):
        """インスタンス変数phrases(リスト）の
           要素('need''phrase'の辞書)
            'need':数値を

            @param mood 現在の機嫌値
        """
        choices = []
        # self.phrasesが保持するリストの要素（辞書）を反復処理する
        for p in self.phrases:
            # self.phrasesの'need'キーの数値と
            # パラメーターmoodをsuitable()に渡す
            # 結果がTrueであればchoicesリストに'phrase'キーの応答例を追加
            if (self.suitable(p['need'], mood)):
                choices.append(p['phrase'])
        # choicesリストが空であればNoneを返す
        if (len(choices) == 0):
            return None
            # choicesリストが空でなければランダムに
            # 応答文字列を選択して返す
        return random.choice(choices)

    def suitable(self, need, mood):
        """必要機嫌値を現在の機嫌値と比較
            
            @param need 必要機嫌値
            @param mood 現在の機嫌値
        """
        # 必要機嫌値が0であればTrueを返す
        if (need == 0):
            return True
        # 必要機嫌値がプラスの場合は機嫌値が必要機嫌値を超えているか判定
        elif (need > 0):
            return (mood > need)
        # 応答例の数値がマイナスの場合は機嫌値が下回っているか判定
        else:
            return (mood < need)

    def add_phrase(self, phrase):
        """インスタンス変数phrases(リスト）の
           要素('need''phrase'の辞書)
            'need':数値を
            
            @param phrase インプット文字列
        """
        # インプット文字列がphrasesリストの応答例に一致するか
        # self.phrases  インプットにマッチした応答フレーズの辞書リスト
        # [ {'need'  : 応答例の整数部分, 'phrase': 応答例の文字列部分}, ... ]
        for p in self.phrases:
            # 既存の応答例に一致したら終了
            if p['phrase'] == phrase:
                return
        # phrasesリストに辞書を追加
        # {'need'  :0, 'phrase':インプット文字列}
        self.phrases.append({'need': 0, 'phrase': phrase})
        
    def make_line(self):
        """

        """
        # 必要機嫌値 + '##' + パターン
        pattern = str(self.modify) + '##' + self.pattern
        phrases= []
        for p in self.phrases:
            resp = str(p['need']) + '##' + str(p['phrase'])
            phrases.append(str(p['need']) + '##' + str(p['phrase']))
        line = pattern + '\t' + '|'.join(phrases)
        return pattern + '\t' + '|'.join(phrases)


