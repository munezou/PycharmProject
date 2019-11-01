class Michael:
    def __init__(self, max = 5, count = 0):
        self.__max = max
        self.__count = count

    def get_max(self):
        return self.__max
    
    def set_max(self, max):
        if max < 5:
            self.__max = 5
        else:
            self.__max = max
        
    def get_count(self):
        return self.__count
    
    def set_count(self, count):
        self.__count = count
    
    max = property(get_max, set_max)
    count = property(get_count, set_count)

    def teach(self):
        if self.count < self.max:
            print('もっと強く！')
        else:
            print('よーしオッケーだ')
        self.count += 1

oni = Michael()
oni.max = 2
for i in range(6):
    print('スマッシュ')
    oni.teach()

