class Michael:
    def __init__(self, max):
        self.max = max
        self.count = 0

    def teach(self):
        if self.count < self.max:
            print('もっと強く！')
        else:
            print('よーしオッケーだ')
        self.count += 1

Michael.final_word = '少し休んでいいぞ!'
def teaching(self):
    i = (self.max) - (self.count)
    if i >= 0:
        print('あと', i + 1, '回!')
    elif i < 0:
        print(self.__class__.final_word)

Michael.teaching = teaching

oni = Michael(5)
for i in range(6):
    print('スマッシュ')
    oni.teach()
    oni.teaching()

