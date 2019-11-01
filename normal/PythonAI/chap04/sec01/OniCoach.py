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

oni = Michael(5)
for i in range(6):
    print('スマッシュ')
    oni.teach()
