import numpy as np

def optimize(X, y, order, fit_func, randomize_order=False):
    order = np.shuffle(order)
    for param_name in order:
            param_set = params[param_name]

            test_accuracies = []
            for param_value in param_set:
                temp_params = base_params.copy()
                temp_params[param_name] = param_value
                # fit to training data (TODO)
                fit_func(X, y, **temp_params)
                # Cross validate score (TODO)

                # Append to test_accuracies the cross validation score (TODO)
                pass

            best_param_idx = np.argmax(test_accuracies)

            # Replace the base_params with the optimal one found by this step
            base_params[param_name] = param_set[best_param_idx]

def gridsearch(X, y, base_params, params, fit_func, order=None, no_init_passes=1, no_random_passes=0):
    """
    fit_func is the function which has a named input signature equal to base_params keys.
    """
    # For each of the parameters
    if order is None:
        order = list(base_params.keys())
    
    # Make a copy of the dictionary because it is going to change
    base_params = base_params.copy()
    
    for _ in range(no_init_passes):
        optimize(X, y, order, fit_func, randomize_order=False)
        
    for _ in range(no_random_passes):
        optimize(X, y, order, fit_func, randomize_order=True)
    
    
    
    