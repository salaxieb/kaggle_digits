## Kaggle digits preprocessing

Some trics for kaggle mnist data recognition competition

### Dependencies:
python3
tensorflow
keras
marplotlib
scipy

### Idea:

![screenshot](screenshots/digits_before.png?raw=true)

Rotate digit to vertical and stretch

![screenshot](screenshots/digits_after.png?raw=true)

For human it makes digits less recognisible, but for NN digits start to look more similar

### Result:

![screenshot](screenshots/results.png?raw=true)

** ~3% improovement compare to same architechture, without preprocessing **

!important: no improovement in case of usage CNN