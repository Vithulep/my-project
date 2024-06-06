import random

class Allocation():
    def __init__(self,old_allocation,new_allocation=None,compliance=None):
        if new_allocation==None:
            self.allocation=old_allocation
        else:
            self.allocation=old_allocation
            for district_id in self.allocation.keys():
                for crop_id in self.allocation[district_id].keys():
                    # print("hi")
                    self.allocation[district_id][crop_id]=(1-compliance)*old_allocation[district_id][crop_id]+compliance*new_allocation[district_id][crop_id]
