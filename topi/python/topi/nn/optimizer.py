from .. import cpp

def SGDOptimizer(weights, gradients, learning_rate):
    return cpp.nn.SGDOptimizer(weights, gradients, learning_rate)

def AdamOptimizer(weights, gradients, learning_rate, t, beta1, beta2, epsilon, ms, vs):
    return cpp.nn.AdamOptimizer(weights, gradients, learning_rate, t, beta1, beta2, epsilon, ms, vs)
