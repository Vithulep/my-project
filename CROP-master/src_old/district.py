from coordinate import Coordinate

class District():
    def __init__(self,id,name,latitude,longitude,population):
        self.id=id
        self.name=name
        self.coordinates=Coordinate(latitude,longitude)
        self.population = population

    def reset(self,crops):
        self.stock={}
        self.production_cost=0
        for crop_id in crops.keys():
            self.stock[crop_id]=None
    
    def get_crop_details(self,crops,i):
        temp = []
        for crop_id,j in zip(crops.keys(),range(len(crops.keys()))):
            temp.append([i,j,crops[crop_id].cost_dict[self.id],crops[crop_id].yield_dict[self.id],crops[crop_id].price_dict[self.id],crops[crop_id].demand_max_dict[self.id],crops[crop_id].demand_min_dict[self.id]])
        return temp
    
    def min_demand_met(self,crops):
        for crop_id in self.stock.keys():
            if round(self.stock[crop_id]) < round(crops[crop_id].demand_min_dict[self.id]):
                return False
        return True

    def max_demand_met(self,crops):
        for crop_id in self.stock.keys():
            if self.stock[crop_id] > crops[crop_id].demand_max_dict[self.id]:
                return False
        return True

    def get_revenue(self,crops):
        revenue=0
        for crop_id in self.stock.keys():
            # revenue+=max(self.stock[crop_id],crops[crop_id].demand_max_dict[self.id])*crops[crop_id].price_dict[self.id]*10 #Earlier
            revenue+= self.stock[crop_id]*crops[crop_id].price_dict[self.id]*10
        return revenue
    
    def get_price_per_ton(self,crops,crop_id):
        # cost1 = []
        # for crop_id in crops.keys():
        #     cost1.append(crops[crop_id].price_dict[self.id]*10)
        ans = crops[crop_id].price_dict[self.id]*10
        return ans

    def is_satisfied(self,crop):
        if self.stock[crop.id]<crop.demand_min_dict[self.id]:
            return False
        return True

    def can_give(self,crop):
        temp = self.stock[crop.id]-crop.demand_min_dict[self.id]
        ans = False
        if temp>0:
            ans = True
        return ans
    
    def amt_give(self,crop):
        return (self.stock[crop.id]-crop.demand_min_dict[self.id])

    def requirement(self,crop):
        return (crop.demand_min_dict[self.id] - self.stock[crop.id])

    def produce(self,allocation,crops):
        self.reset(crops)
        crop_allocation=allocation.allocation[self.id]
        for crop in crops.values():
            self.stock[crop.id]=crop.yield_dict[self.id]*crop_allocation[crop.id]
            self.production_cost+=crop_allocation[crop.id]*crop.cost_dict[self.id]

    def can_take(self,crop):  
        temp = crop.demand_max_dict[self.id] - self.stock[crop.id]
        ans = False
        if temp>0:
            ans = True
        return ans
    
    def can_receive(self,crop): #0->receiver, 1->donor, 2-> both
        temp = 0
        if self.stock[crop.id] < crop.demand_min_dict[self.id]:
            temp = 0
        elif self.stock[crop.id] > crop.demand_max_dict[self.id]:
            temp = 2 
        else:
            temp = 1
        return temp
    
