from causaldmir.discovery.funtional_based.anm.ANM import ANM
from causaldmir.discovery.funtional_based.anm.ANM import FastHsicTestGamma
import numpy as np
import pandas as pd
from unittest import TestCase,main


class TestANM(TestCase):


    def test_anm_using_simulation(self):
        n_runs = 10
        p_value_forward_list = []
        p_value_backward_list = []
        
        for _ in range(n_runs):
            data_x, data_y = self.simulate_data()
            anm = ANM()
            p_value_forward, p_value_backward = anm.cause_or_effect(data_x, data_y)
            p_value_forward_list.append(p_value_forward)
            p_value_backward_list.append(p_value_backward)

        avg_p_value_forward = np.mean(p_value_forward_list)
        avg_p_value_backward = np.mean(p_value_backward_list)
        
        print(f"Average p_value_forward: {avg_p_value_forward}")
        print(f"Average p_value_backward: {avg_p_value_backward}")

        self.assertLess(avg_p_value_forward, avg_p_value_backward, 
                           f"Average forward p-value ({avg_p_value_forward}) is not greater than average backward p-value ({avg_p_value_backward})")


    def simulate_data(self):
        data_x = np.random.normal(size=100)
        data_y = np.sin(data_x) + 0.1 * np.random.normal(size=100)
    
        return data_x, data_y
    

def test_hsic():
    x = np.random.normal(size=100)
    y = np.random.normal(size=100)
    score = FastHsicTestGamma(x, y)
    
    print(f"HSIC score: {score}")
    assert score >= 0, "HSIC score should be non-negative"

    if __name__ == '__main__':
       main()
       test_hsic()