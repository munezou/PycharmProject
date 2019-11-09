'''
------------------------------------------------------------------------------------------------------------------------
7.5 clouser
------------------------------------------------------------------------------------------------------------------------
'''

print('-----------------------------------------------------------------------------------------------------------------\n'
      '                                              7.5 clouser                                                        \n'
      '-----------------------------------------------------------------------------------------------------------------\n')

print('---< Class that calculates the average value of gradually added data >---')
# Class that calculates the average value of gradually added data
class Averager():
    def __init__(self):
        self.series = []

    def __call__(self, new_value):
        self.series.append(new_value)
        total = sum(self.series)
        return total / len(self.series)

avg = Averager()
print('avg(10) = {0}'.format(avg(10)))
print('avg(11) = {0}'.format(avg(11)))
print('avg(12) = {0}'.format(avg(12)))
print()

print('---< Higher order function that calculates the average value of gradually added data >---')
def make_averager():
    series = []

    def avarager(new_value):
        series.append(new_value)
        total = sum(series)
        return total / len(series)

    return avarager

avg = make_averager()
print('avg(10) = {0}'.format(avg(10)))
print('avg(11) = {0}'.format(avg(11)))
print('avg(12) = {0}'.format(avg(12)))
print()

print('avg.__code__.co_varnames = {0}'.format(avg.__code__.co_varnames))
print()
print('avg.__code__.co_freevars = {0}'.format(avg.__code__.co_freevars))
print()
print('avg.__closure__ = {0}'.format(avg.__closure__))
print()
print('avg.__closure__[0].cell_contents = {0}'.format(avg.__closure__[0].cell_contents))
print()


