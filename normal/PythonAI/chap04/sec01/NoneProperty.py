class Michael:
    def __init__(self, max = 5, count = 0):
        self.max = max
        self.count = count

    def teach(self):
        if self.count < self.max:
            print('もっと強く！')
        else:
            print('よーしオッケーだ')
        self.count += 1

oni = Michael()
oni.count = 1

for i in range(5):
    print('スマッシュ')
    oni.teach()
    
