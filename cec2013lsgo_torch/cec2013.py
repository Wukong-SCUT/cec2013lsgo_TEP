from F1 import F1 as f1
from F2 import F2 as f2
from F3 import F3 as f3
from F4 import F4 as f4
from F5 import F5 as f5
from F6 import F6 as f6
from F7 import F7 as f7
from F8 import F8 as f8
from F9 import F9 as f9
from F10 import F10 as f10
from F11 import F11 as f11
from F12 import F12 as f12
from F13 import F13 as f13
from F14 import F14 as f14
from F15 import F15 as f15

class Benchmark():
    def __init__(self, device):
        self.device = device
    def get_function(self, func_id):
        if func_id == 1:
            return f1(self.device)
        elif func_id == 2:
            return f2(self.device)
        elif func_id == 3:
            return f3(self.device)
        elif func_id == 4:
            return f4(self.device)
        elif func_id == 5:
            return f5(self.device)
        elif func_id == 6:
            return f6(self.device)
        elif func_id == 7:
            return f7(self.device)
        elif func_id == 8:
            return f8(self.device)
        elif func_id == 9:
            return f9(self.device)
        elif func_id == 10:
            return f10(self.device)
        elif func_id == 11:
            return f11(self.device)
        elif func_id == 12:
            return f12(self.device)
        elif func_id == 13:
            return f13(self.device)
        elif func_id == 14:
            return f14(self.device)
        elif func_id == 15:
            return f15(self.device)
        else:
            raise ValueError("Function id is out of range.")

    def get_info(self, func_id):
        if func_id == 1:
            f1_ = f1(self.device)
            return f1_.info()
        elif func_id == 2:
            f2_ = f2(self.device)
            return f2_.info()
        elif func_id == 3:
            f3_ = f3(self.device)
            return f3_.info()
        elif func_id == 4:
            f4_ = f4(self.device)
            return f4_.info()
        elif func_id == 5:
            f5_ = f5(self.device)
            return f5_.info()
        elif func_id == 6:
            f6_ = f6(self.device)
            return f6_.info()
        elif func_id == 7:
            f7_ = f7(self.device)
            return f7_.info()
        elif func_id == 8:
            f8_ = f8(self.device)
            return f8_.info()
        elif func_id == 9:
            f9_ = f9(self.device)
            return f9_.info()
        elif func_id == 10:
            f10_ = f10(self.device)
            return f10_.info()
        elif func_id == 11:
            f11_ = f11(self.device)
            return f11_.info()
        elif func_id == 12:
            f12_ = f12(self.device)
            return f12_.info()
        elif func_id == 13:
            f13_ = f13(self.device)
            return f13_.info()
        elif func_id == 14:
            f14_ = f14(self.device)
            return f14_.info()
        elif func_id == 15:
            f15_ = f15(self.device)
            return f15_.info()
        else:
            raise ValueError("Function id is out of range.")
    
    def get_num_functions(self):
        return 15



