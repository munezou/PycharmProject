import random

stroke = 'ストローク'
volley = 'ボレー'
smash = 'スマッシュ'
lob = 'ロブ'
for count in range(5):
    x = random.randint(1, 10)
    if x <= 3:
        print(volley)
    elif x >= 4 and x <= 5:
        print(smash)
    elif x >= 6 and x <= 7:
        print(lob)
    else:
        print(stroke)
        
