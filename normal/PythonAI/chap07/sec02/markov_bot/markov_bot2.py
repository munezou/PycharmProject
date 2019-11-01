import re
import random
from analyzer import *
from itertools import chain

class Markov:
    def make(self):
        """ マルコフ連鎖を利用して文章を作り出す
        """
        print('テキストを読み込んでいます...')
        filename = "bocchan.txt"
        with open(filename, "r", encoding = 'utf_8') as f:
            text = f.read()
        text = re.sub("\n","", text)
        wordlist = parse(text)

	# マルコフ辞書の作成
        markov = {}
        p1 = ''
        p2 = ''
        p3 = ''
        for word in wordlist:
    	    # p1、p2、p3のすべてに値が格納されているか
            if p1 and p2 and p3:
       		# markovに(p1, p2, p3)キーが存在するか
                if (p1, p2, p3) not in markov:
            	# なければキー：値のペアを追加
                    markov[(p1, p2, p3)] = []
        	# キーのリストにサフィックスを追加（重複あり）
                markov[(p1, p2, p3)].append(word)
            # 3つのプレフィックスの値を置き換える
            p1, p2, p3 = p2, p3, word

        # マルコフ辞書から文章を作り出す
        count = 0
        sentence = ''
    	# markovのキーをランダムに抽出し、プレフィックス1〜3に代入
        p1, p2, p3  = random.choice(list(markov.keys()))
        #while count < 30:
        while count < len(wordlist):
            # キーが存在するかチェック
            if ((p1, p2, p3) in markov) == True:
                # 文章にする単語を取得
                tmp = random.choice(markov[(p1, p2, p3)])
                # 取得した単語をsentenceに追加
                sentence += tmp
            # 3つのプレフィックスの値を置き換える
            p1, p2, p3 = p2, p3, tmp
            count += 1

        # 最初に出てくる句点(。)までを取り除く
        sentence = re.sub("^.+?。", "", sentence)
        # 最後の句点(。)から先を取り除く
        if re.search('.+。', sentence):
            sentence = re.search('.+。', sentence).group()
        # 閉じ括弧を削除
        sentence = re.sub("」", "", sentence)
        #開き括弧を削除
        sentence = re.sub("「", "", sentence)
        #全角スペースを削除
        sentence = re.sub("　", "", sentence)

        # 生成した文章を戻り値として返す
        return sentence




#=================================================
# プログラムの起点
#=================================================
if __name__  == '__main__':
    # Markovオブジェクトを生成
    markov = Markov()
    # マルコフ連鎖で生成された文章群を取得
    text = markov.make()
    # 各文章の末尾の改行で分割してリストに格納
    sentences = text.split('。')
    # リストから空の要素を取り除く
    if '' in sentences:
        sentences.remove('')
    print ("会話をはじめましょう。")

    while True:
        line = input(' > ')
        # インプット文字列を形態素解析
        parts = analyze(line)

        m = [] #
        # 解析結果の形態素と品詞に対して反復処理
        for word, part in parts:
            #print('word===',word)
            #print('part===',part)
            
            # インプット文字列に名詞があればそれを含むマルコフ連鎖文を検索
            if keyword_check(part):
                #print('afetr_check_word===',word)
                # マルコフ連鎖で生成した文章を1つずつ処理
                for element in sentences:
                    # 形態素の文字列がマルコフ連鎖の文章に含まれているか検索する
                    # 最後を'.*?'にすると「花」のように検索文字列だけにもマッチするので
                    # + '.*'として検索文字列だけにマッチしないようにする
                    find = '.*?' + word + '.*'
                    # マルコフ連鎖文にマッチさせる
                    tmp = re.findall(find, element)
                    if tmp:
                        # マッチする文章があればリストmに追加
                        m.append(tmp)
        # findall()はリストを返してくるので多重リストをフラットにする
        m = list(chain.from_iterable(m))
                    
        if m:
            # インプット文字列の名詞にマッチしたマルコフ連鎖文からランダムに選択
            print(random.choice(m))
        else:
            # マッチするマルコフ連鎖文がない場合
            print(random.choice(sentences))

