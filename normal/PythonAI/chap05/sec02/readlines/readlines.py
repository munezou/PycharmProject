file = open('dics/random.txt', 'r',encoding = 'utf_8')
lines = file.readlines() # 1行ずつ読み込む(各要素の末尾に改行文字が追加される)
file.close()
for line in lines:
    print (line)
