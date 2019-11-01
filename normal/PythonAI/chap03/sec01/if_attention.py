smash = 'スマッシュ'
stroke = 'ストローク'
volley = 'ボレー'
for count in range(10):
    if (count % 2 == 0):
        print(volley)
    elif (6 <= count) and ( count % 2 == 0):
        print(smash)
    else:
        print(stroke)
