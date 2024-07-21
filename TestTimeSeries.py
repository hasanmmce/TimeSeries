import numpy as np
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# check all above imported libraries working properly
def test_imports():
    assert np.array([1, 2, 3]).shape == (3,)
    assert pd.Series([1, 2, 3]).shape == (3,)
    assert LinearRegression().fit(np.array([1, 2, 3]).reshape(-1, 1), np.array([1, 2, 3])).coef_.shape == (1,)
    print("All libraries imported successfully!")


#try the test_imports function
test_imports()

#import rpy2 and test that the imported library is working properly
import rpy2
print("version",rpy2.__version__)

import rpy2.robjects as ro

# Basic R command
ro.r('x <- c(1, 2, 3, 4, 5)')
ro.r('print(mean(x))')  # Should print the mean of the vector x


# Create a Python list and convert it to an R vector
py_list = [1, 2, 3, 4, 5]
r_vector = ro.FloatVector(py_list)
print(r_vector)

# Create a Python NumPy array and convert it to an R matrix
py_array = np.array([[1, 2, 3], [4, 5, 6]])
r_matrix = ro.r.matrix(ro.FloatVector(py_array.flatten()), nrow=2)
print(r_matrix)





