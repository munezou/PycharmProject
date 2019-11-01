print('身長(cm)は？')
height = float(input('you > '))
bmi = 22  # BMI
weight = bmi * (height / 100) ** 2
print('身長が' + str(height) + 'cmの場合の標準体重は', end='')
print('{:.2f}kgです。'.format(weight))
