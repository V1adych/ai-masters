import time
import numpy as np


class GradientDescent:
    def __init__(self, StepSizeChoice, return_history=True, name=None):
        self.name = name
        self.StepSizeChoice = StepSizeChoice
        self.return_history = return_history
        self.history = []
    
    def __call__(self, x0, f, gradf, N):
        self.history = [(x0, time.time())]
        x = x0.copy()
        for k in range(N):
            h = -gradf(x)
            alpha = self.StepSizeChoice(x, h, k, gradf, f)
            x = x + alpha * h
            if self.return_history:
                self.history.append((x, time.time()))
        return x
    
    def solve(self, x0, f, gradf, tol=1e-3, max_iter=10000):
        self.history = [(x0, time.time())]
        x = x0.copy()
        k = 0
        x_prev = None
        while x_prev is None or np.linalg.norm(gradf(x)) > tol: 
            h = -gradf(x)
            alpha = self.StepSizeChoice(x, h, k, gradf, f)
            x_prev, x = x, x + alpha * h
            if self.return_history:
                self.history.append((x, time.time()))
            if k >= max_iter:
                break
            k += 1
        return x

class CG_FR:
    def __init__(self, StepSizeChoice, restart=False, return_history=True, name=None):
        self.name = name
        self.StepSizeChoice = StepSizeChoice
        self.return_history = return_history
        self.restart = restart
        self.history = []
    
    def solve(self, x0, f, gradf, tol=1e-3, max_iter=10000):
        self.history = [(x0, time.time())]
        x = x0.copy()
        k = 0
        grad = gradf(x)
        p = -grad
        while k == 0 or np.linalg.norm(gradf(x)) > tol: 
            alpha = self.StepSizeChoice(x, p, k, gradf, f)
            if alpha < 1e-18:
                break
            x = x + alpha * p
            grad_next = gradf(x)
            beta = grad_next.dot(grad_next) / grad.dot(grad)
            p = -grad_next + beta * p
            grad = grad_next.copy()
            k += 1
            if self.restart and k % self.restart == 0:
                grad = gradf(x)
                p = -grad
            if self.return_history:
                self.history.append((x, time.time()))
            if k >= max_iter:
                break
        return x

    
class CG_PR:
    def __init__(self, StepSizeChoice, restart=False, return_history=True, name=None):
        self.name = name
        self.StepSizeChoice = StepSizeChoice
        self.return_history = return_history
        self.restart = restart
        self.history = []
    
    def solve(self, x0, f, gradf, tol=1e-3, max_iter=10000):
        self.history = [(x0, time.time())]
        x = x0.copy()
        k = 0
        grad = gradf(x)
        p = -grad
        while k == 0 or np.linalg.norm(gradf(x)) > tol: 
            alpha = self.StepSizeChoice(x, p, k, gradf, f)
            if alpha < 1e-18:
                break
            x = x + alpha * p
            grad_next = gradf(x)
            beta = grad_next.dot(grad_next-grad) / grad.dot(grad)
            p = -grad_next + beta * p
            grad = grad_next.copy()
            k += 1
            if self.restart and k % self.restart == 0:
                grad = gradf(x)
                p = -grad
            if self.return_history:
                self.history.append((x, time.time()))
            if k >= max_iter:
                break
        return x

def parse_logs(xhistory, ret_time=False, funcx=None):
    values = [funcx(x) for x, _ in xhistory]
    if ret_time:
        times = [t for _, t in xhistory]
        times = [times[ind]-times[0] for ind, t in enumerate(times)]
    else:
        times = [i for i in range(len(xhistory))]
    return times, values
