import numpy as np
import sigmoid
def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost 
    """

    m, n = X.shape
    
    cost = 0
    for x_i, y_i in zip(X, y):
#         print(x_i, y_i)
        loss = 0
        z = np.dot(w, x_i) + b
#         print("z", z)
        f_x = sigmoid(z)
#         print(f_x)
        if y_i == 0:    
            """
            I veer from our esteemed Professor's combination of the two possible terms, 
            as I believe that only evaluating the relevant term will improve computation speed. 
            """
            loss = -np.log(1 - f_x)
        else:
            loss = -np.log(f_x)
#         print("loss", loss)
        cost += loss
    
    cost /= m
    print("cost", cost) 
        
        
    total_cost = cost  

    return total_cost