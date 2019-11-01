class Dictionary:
    def __init__(self):
        self.random = [] # (1)
        # ランダム辞書ファイルオープン
        rfile = open('dics/random.txt', 'r', encoding = 'utf_8')
        # 各行を要素としてリストに格納
        r_lines = rfile.readlines()
        rfile.close()

        # 末尾の改行と空白文字を取り除いて
        # インスタンス変数（リスト）に格納
        self.random = []
        for line in r_lines:
            str = line.rstrip('\n')
            if (str!=''):
                self.random.append(str)

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
        # 辞書型のインスタンス変数を用意
        self.pattern = {}
        # 1行をタブで切り分けて
        # 'pattern'キー：正規表現のパターン
        # 'phrases'キー：応答例
        for line in self.new_lines:
            p_r = line.split('\t')
            self.pattern.setdefault('pattern', []).append(p_r[0])
            self.pattern.setdefault('phrases', []).append(p_r[-1])
