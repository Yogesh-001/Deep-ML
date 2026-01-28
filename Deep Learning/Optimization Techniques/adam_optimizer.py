import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
	# Your code here
	x = x0.copy()
    m_t = np.zeros_like(x)
    v_t = np.zeros_like(x)

    for t in range(1 , num_iterations + 1):
        g_t = grad(x)

        m_t = beta1 * m_t + (1-beta1)*g_t
        v_t = beta2 * v_t + (1-beta2) * (g_t**2)

        m_hat = m_t/(1-beta1 ** t)
        v_hat = v_t/(1-beta2 ** t)

        x = x - learning_rate * m_hat/(np.sqrt(v_hat) +  epsilon)
    
    return x
