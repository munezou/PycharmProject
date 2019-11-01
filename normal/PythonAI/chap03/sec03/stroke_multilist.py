selves = ['ロビング', 'ネット&ボレー', 'クロスショット', 'トップスピン']
others = ['サービス', 'リターン[ボレー]', 'リターン[ストローク]', 'リターン[スマッシュ]']

n = min(len(selves), len(others))
for i in range(n):
     print (others[i], selves[-i-1], sep=' <-- ')
