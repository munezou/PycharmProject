name = input('お名前は? >')     # 名前を取得
print('%s さん、こんにちは!' % name)
prompt = name + ' > '           # プロンプトを作る
while 1:
    answer = input(prompt)      # 入力された文字列を取得
    if answer == 'さよなら':    # 'さよなら'でループ終了
        print('バイバイ')
        break
    elif not answer:            # 未入力ならループ終了
        print('......')
        break
    print('「{}」なんですね。'.format(answer))    # 入力された文字列を表示
