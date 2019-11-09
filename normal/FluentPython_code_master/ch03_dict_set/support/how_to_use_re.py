import re

'''
------------------------------------------------------
re.sub(pattern, repl, string, count=0, flags=0)
------------------------------------------------------
aruguments)
 pattern: regular expression pattern
 repl: String to replace
 string: String to be replaced
 count: Number of replacements
 count and flags are optional
'''

print ('---< Sample code containing a period >---')
text = 'HelloTWorld'
pattern = r'Hello.World'    # . Matches any single character.

result = re.sub(pattern,'is replaced',text)
print(text + ' ' + result) # Result: HelloTWorld is replaced

print()


print ('---< Sample code including exponentiation >---')

text = 'Hello'
pattern = r'^Hello'        # ^ Match the beginning of the line.

result = re.sub(pattern,'is replaced',text)
print(text + ' ' + result) # Result: Hello is replaced
print()

print('---< Sample code containing dollar signs >---')

text = 'Hello'
pattern = r'Hello$'     # $ Match the end of the line.

result = re.sub(pattern,'is replaced',text)
print(text + ' ' + result) # Result: Hello is replaced
print()

print ('---< Sample code containing asterisks >---')
text = 'Hello'
pattern = r'Hel*o'      # The * following a single character matches zero or more repetitions of the expression.

result = re.sub(pattern,'is replaced',text)
print(text + ' ' + result) # Result: Hello is replaced
print()

print ('---< Sample code containing plus >---')

text = 'Hello'
'''
A + followed by a letter matches one or more occurrences of the expression.
The only difference from the asterisk is zero or more or one or more.
'''
pattern = r'Hel+o'

result = re.sub(pattern,'is replaced',text)
print(text + ' ' + result) #Œ‹‰Ê : Hello is replaced
print()

print ('---< Sample code containing a question mark >---')

text = 'Hello World'
pattern = r'Hello ?World'   # The question mark matches when the preceding expression is zero or one.

result = re.sub(pattern,'is replaced',text)
print(text + ' ' + result) # Result: Hello World is replaced

print()

print ('---< Sample code containing {m} >---')

text = 'Hello'
pattern = r'Hel{2}o'    # {m} Match m iterations of the previous group.

result = re.sub(pattern,'is replaced',text)
print(text + ' ' + result)  # Result : Hello is replaced
print()

print ('---< Sample code containing {m, n} >---')
text = 'Hello'
pattern = r'Hel{1,3}o'      # {m, n} Matches m to n repetitions of the previous group.

result = re.sub(pattern,'is replaced',text)
print(text + ' ' + result) # Result : Hello is replaced
print()

print ('---< Sample code containing () >---')
text = 'HeHeo'
pattern = r'(He){2}o'   # () can group strings.

result = re.sub(pattern,'is replaced',text)
print(text + ' ' + result) # Result : HeHeo is replaced
print()

print ('---< Sample code containing [] >---')
text = 'Hello'
pattern = r'He[el]lo'   # [] Match a single character contained within parentheses.

result = re.sub(pattern,'is replaced',text)
print(text + ' ' + result) # Result : Hello is replaced
print()

print ('---< Sample code containing | >---')
text = 'Hello World'
pattern = r'Hello|test|a'   # | Vertical bars match choices.

result = re.sub(pattern,'is replaced',text)
print(text + ' ' + result) # Result : Hello World is replaced World


