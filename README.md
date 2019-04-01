## Kaggle digits preprocessing

Some trics for kaggle mnist data recognition competition

### Dependencies:

* python3
* tensorflow
* keras
* marplotlib
* scipy

### Architechture

**Input:** (60000, 784, 1) (60000 images shape = 28x28)

**Hidden:** Iteratively changing hidden layer size

**Output:** One hot vector with 10 labels (0..9)

![screenshot](screenshots/mlp.png?raw=true)

### Idea:

![screenshot](screenshots/digits_before.png?raw=true)

Rotate digit to vertical and stretch


![screenshot](screenshots/digits_after.png?raw=true)

For human it makes digits less recognisible, but for NN digits start to look more similar

### Result:

![screenshot](screenshots/results.png?raw=true)

## **~5% improovement compare to same architechture, without preprocessing**

!important: no improovement in case of usage CNN