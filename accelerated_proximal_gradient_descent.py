import numpy as np

def accelerated_proximal_gradient_descent(X, y, initial_val, arg, 
                                          partial_obj, partial_gradient, prox, 
                                          step_shrinkage, initial_step,  
                                          no_iters = 1e+3, tolerance = 1e-8):
    
    step = initial_step
    # stack previous two values
    all_val = np.vstack( (initial_val, initial_val) )
    all_val = np.vstack( (all_val, initial_val) )
    
    it = 1
    current_val = initial_val
    
    def norm(z):
        return sqrt(sum(np.square(z)))
    
    def stopping_rule():
        return (norm(all_val[-1] - all_val[-2]) < tolerance * max(1, norm(all_val[-1])))
    
    def more_curvature_than_quadratic(gg, weight, step):
        potential_new_obj   = partial_obj(X, y, weight - step * gg, **arg) 
        obj_minus_quadratic = partial_obj(X, y, weight, **arg) 
        - step * gradient(X, y, weight, **arg).dot(gg) + step / 2.0 * norm(gg)
        return (potential_new_obj > obj_minus_quadratic) 
    
    while True:
        print 'iter ', it
        
        # check for shrinking
        to_shrink_step = True
        while to_shrink_step:
            # perform intermediate update to carry some "momentum"
            inter_val = current_val + (it - 2.0) / (it + 1.0) * (all_val[-2] - all_val[-1])
            new_val   = inter_val - step * partial_gradient(X, y, inter_val, **arg)
            generalized_gradient = ( inter_val - prox(X, y, new_val, step, **arg) ) / step
            if more_curvature_than_quadratic(generalized_gradient, inter_val, step):
                step = step_shrinkage * step
            else:
                to_shrink_step = False
        
        # perform update 
        print 'step ', step
        current_val = inter_val - step * generalized_gradient
        all_val = np.vstack( (all_val, current_val) )
        
        if (it == no_iters) or (stopping_rule()):
            return all_val
    
        it += 1
        print
    
    return all_val    

"""
example of usage:

lasso problem 

partial_obj = lambda X, y, w, alpha: 0.5 * np.sum(np.square( y - X.dot(w) ))
obj = lambda X, y, w, alpha: partial_obj(X, y, w, alpha)  + alpha * sum(abs(w)) 
partial_gradient = lambda X, y, w, alpha: -X.T.dot( y - X.dot(w) )
def prox(X, y, w, t, alpha): 
    result = np.zeros(len(w))
    alpha = t * alpha
    for i, w_ in enumerate(w):
        if w_ > alpha:
            result[i] = w_ - alpha
        elif w_ < -alpha:
            result[i] = w_ + alpha
        else:
            result[i] = 0.0
    return result

acc_prox_w = accelerated_proximal_gradient_descent(X = X, y = y, initial_val = np.zeros(500), 
                                  arg = {'alpha': alpha}, 
                                  partial_obj = partial_obj, partial_gradient = partial_gradient, 
                                  prox = prox, step_shrinkage = 0.9, initial_step = 1.0,
                                  no_iters = 1e+3, tolerance = 1e-4)

"""
