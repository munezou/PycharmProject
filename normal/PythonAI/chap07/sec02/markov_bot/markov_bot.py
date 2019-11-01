import re
import random
from analyzer import *


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
    markov = Markov()
    text = markov.make()
    ans = text.split('。')
    if '' in ans:
        ans.remove('')
    print ('会話をはじめましょう。')

    while True:
        message = input('>')
        if ans:
            print(random.choice(ans))

