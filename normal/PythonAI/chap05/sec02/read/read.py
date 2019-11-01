file = open('dics/random.txt', 'r', encoding = 'utf_8')
data = file.read()  # ファイル終端まで読み込んでデータを返す
file.close()

lines = data.split('\n') # 改行で区切る(改行文字そのものは戻り値のデータには含まれない)
for line in lines:
    print(line)
