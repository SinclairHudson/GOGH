from os import listdir
import shutil

count = 0
for filename in listdir('./trainset/colourGogh'):
    if count % 7 == 0:
        shutil.move('./trainset/colourGogh/'+filename, './testset/colourGogh/'+filename)

    count = count + 1

count = 0
for filename in listdir('./trainset/openImages'):
    if count % 7 == 0:
        shutil.move('./trainset/openImages/' + filename, './testset/openImages/' + filename)

    count = count + 1