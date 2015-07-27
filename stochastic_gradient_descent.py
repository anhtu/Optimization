import numpy as np

def stochastic_gradient_descent(gradient, arg, size, no_var, initial_val, 
                                learning_rate = 0.01, tolerance = 0.001, 
                                no_iterations = 1000):
    """ Perform gradient descent on a given (convex) objective function f, the gradient of the
    function is provided as argument to the method, as well as the learning rate alpha. 
    The gradient is a vector of partial derivatives, here the gradient is the gradient of one example 
    tolerance: when to stop
    no_var: number of variables 
    arg: list of arguments to pass to gradient 
    
    """
    if type(initial_val) == list and len(initial_val) == 0:
        initial_val = np.zeros(no_var)
    
    if type(initial_val) == list:
        initial_val = np.array(initial_val)
    
    # check if training examples is passed
    if type(arg) == dict:
        if len(arg.values() ) < 2:
            print "Please supply training examples"
            return
    else:
        print "Please supply training examples"
        return
    
    def stochastic_gradient(w, x, y, grad_one_example, size=1):
        # randomly select without replacement from x, y
        # default as stochastic - single example
        
        selected = np.random.choice(xrange(len(x)), size=size, replace=False)
        x_sub, y_sub = x[ [selected] ], y[ [selected] ]
        grad = np.zeros( len(w) )
        for (x_, y_) in zip(x_sub, y_sub):
            grad = grad + gradient_one_example(w, x_, y_)
    
        return grad
    
    arg['w'] = initial_val
    arg['grad_one_example'] = gradient
    if size != None:
        arg['size'] = size
    current_gradient = stochastic_gradient(**arg)
    new_var = initial_val
    it = 0   # #iterations

    while abs(np.max(current_gradient)) > tolerance:
        # update
        new_var = new_var - learning_rate * current_gradient
        arg['w'] = new_var
        current_gradient = stochastic_gradient(**arg)
        it += 1
        if (it == no_iterations):
            return new_var, no_iterations
        
    return new_var, it

"""
example of usage: 
def gradient_one_example(w, x, y):
    return np.array([2*(w[0] + w[1]*x - y), 
                     2*( x*w[0] - x*y + w[1]*x**2) ])

stochastic_gradient_descent(gradient_one_example, arg = {'x': x, 'y': y}, size = 1,
                            no_var = 2, 
                            initial_val = [5.0, 5.0],  
                            learning_rate = 0.01, tolerance = 1e-5, no_iterations = 1e+5)
"""