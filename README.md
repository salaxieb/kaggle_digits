## Kaggle digits preprocessing

Some trics for kaggle mnist data recognition competition

### Dependencies:
python3

'''
	tensorflow
	keras
	marplotlib
	scipy
'''

###Idea:
![screenshot](screenshots/digits_before.png?raw=true)

we rotate each image to vertical and stretch

![screenshot](screenshots/digits_after.png?raw=true)

for human it makes digits less recognisible, but for NN digits start to look more similar

### Result:
![screenshot](screenshots/digits_after.png?raw=true)

~3% improove compare to same architechture, but without preprocessing
!important: no improovement in case of usage CNN