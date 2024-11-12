import numpy as np
import sigmoid
def compute_gradient(X, y, w, b, *argv): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_w = np.zeros(w.shape)
    dj_b = 0.

    for i in range(m):
        z_wb = np.dot(X[i], w) + b    # I will use the parallel processing way
        f_wb = sigmoid(z_wb)          # Belongs here
        err_i = f_wb - y[i]
        # Will not be using first loop. I will also skip its' associated follow-up code.
#         for j in range(n):          
#             z_wb += np.dot(w, X[i]) + b
#             print("z_wb", z_wb)
#         z_wb += None
#         dj_db_i = f_wb - y[i]
#         print("dj_db_i", dj_db_i)
#         dj_db += dj_db_i
        
        for j in range(n):
            dj_w[j] += err_i * X[i, j]
            
        dj_b += err_i
        
    dj_dw = dj_w/m
    dj_db = dj_b/m

        
    return dj_db, dj_dw