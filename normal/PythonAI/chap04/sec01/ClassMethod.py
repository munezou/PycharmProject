class Michael:
    count = 0
    max = 5

    @classmethod
    def teach(cls):
        if cls.count < cls.max:
            print('もっと強く！')
        else:
            print('よーしオッケーだ')
        cls.count += 1

for i in range(4):
    print('スマッシュ')
    Michael.teach()
for i in range(2):
    print('バックハンドストローク')
    Michael.teach()
