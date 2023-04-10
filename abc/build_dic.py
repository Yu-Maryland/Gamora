import re
import numpy as np
filename = 'cells.txt'
f = open(filename, 'r')
content_list = []
for line in f.readlines():
    content = re.findall(r'[(](.*?)[)]', line)
    #print(content)
    content_list.append(content)

content_array = np.array(content_list)
print(content_array)
np.savetxt('lib_dict.txt',content_array, delimiter = ',', fmt = '%s')
 
