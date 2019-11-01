import random

sound = {
    'グランドストローク' : '「ポッコーン」',
    'スマッシュ' : '「パコンッ」',
    'ボレー' : '「ベキィッ」'
    }
for count in range(5):
    x = random.randint(1, 10)
    if x <= 4:
        attack = 'グランドストローク'
    elif x >= 4 and x <= 6:
        attack = 'ボレー'
    else:
        attack = 'スマッシュ'
    print(attack, '\n', sound[attack])
