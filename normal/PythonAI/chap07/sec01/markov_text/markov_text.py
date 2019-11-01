from janome.tokenizer import Tokenizer
import re
import random

def parse(text):
    """ 形態素解析によって形態素を取り出す

        @param text マルコフ辞書のもとになるテキスト
        戻り値 形態素のリスト
    """
    t = Tokenizer()# Tokenizerオブジェクトを生成
    tokens = t.tokenize(text)# 形態素解析を実行
    result = []# 形態素を格納するリスト
    for token in tokens:
        result.append(token.surface)
    return(result)

# マルコフ辞書のもとになるテキストファイルを指定
filename = "text.txt"
with open(filename, "r", encoding = 'utf_8') as f:
    text = f.read()
# 文末の改行文字を取り除く
text = re.sub("\n","", text)
# 形態素の部分をリストとして取得
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

# 生成した文章を格納するグローバル変数
sentence = ''
def generate():
    """ マルコフ辞書から文章を作り出す
    """
    global sentence
    # markovのキーをランダムに抽出し、プレフィックス1〜3に代入
    p1, p2, p3  = random.choice(list(markov.keys()))
    # 単語リストの単語の数だけ繰り返す
    count = 0
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
    sentence = re.sub('^.+?。', '', sentence)
    # 最後の句点(。)から先を取り除く
    if re.search('.+。', sentence):
        sentence = re.search('.+。', sentence).group()
    # 閉じ括弧を削除
    sentence = re.sub('」', '', sentence)
    #開き括弧を削除
    sentence = re.sub('「', '', sentence)
    #全角スペースを削除
    sentence = re.sub('　', '', sentence)

def overlap():
    """ 重複した文章を取り除く
    """
    global sentence
    sentence = sentence.split('。')
    if '' in sentence:
        sentence.remove('')
    new = []
    for str in sentence:
        str = str + '。'
        if str=='。':
            break
        new.append(str)
    new = set(new)
    sentence=''.join(new)

while(not sentence):
    generate()
    overlap()


#=================================================
# 生成した文章の出力
#=================================================
if __name__  == '__main__':
    print(sentence)
