
import os
with open("synset.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

for c in content:
  os.system('wget {0}'.format(c))
