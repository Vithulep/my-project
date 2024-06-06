class Transport_Strategy():
    def __init__(self,districts,crops):
        self.reset(districts,crops)

    def reset(self,districts,crops):
        self.cost=0
        self.logs=[]
        self.unsatisfied = []

        for district in districts.values():
            for crop in crops.values():
                    self.unsatisfied.append((crop.id,district.id))

    def is_solution(self,districts,crops):
        for district in districts.values():
            if not district.min_demand_met(crops):
                return False
            # if not district.max_demand_met(crops):
            #     return False
        return True

    def solution_feasible(self,districts,crops):
        total_stock={}
        total_min_demand={}
        feasibility={}
        for district in districts.values():
            for crop in crops.values():
                if crop.id in total_stock.keys():
                    total_stock[crop.id]+=district.stock[crop.id]
                else:
                    total_stock[crop.id]=district.stock[crop.id]

                if crop.id in total_min_demand.keys():
                    total_min_demand[crop.id]+=crop.demand_min_dict[district.id]
                else:
                    total_min_demand[crop.id]=crop.demand_min_dict[district.id]

        flag=True
        for crop_id in total_stock.keys():
            if total_stock[crop_id]<total_min_demand[crop_id]:
                feasibility[crop_id]=False
                flag=False
            else:
                feasibility[crop_id]=True

        return flag,feasibility

    def upper_feasible(self,districts,crops):
        total_stock={}
        total_max_demand={}
        feasibility={}
        for district in districts.values():
            for crop in crops.values():
                if crop.id in total_stock.keys():
                    total_stock[crop.id]+=district.stock[crop.id]
                else:
                    total_stock[crop.id]=district.stock[crop.id]

                if crop.id in total_max_demand.keys():
                    total_max_demand[crop.id]+=crop.demand_max_dict[district.id]
                else:
                    total_max_demand[crop.id]=crop.demand_max_dict[district.id]

        flag=True
        for crop_id in total_stock.keys():
            if total_stock[crop_id]>total_max_demand[crop_id]:
                feasibility[crop_id]=False
                flag=False
            else:
                feasibility[crop_id]=True

        return flag,feasibility

    def feasible(self,d1,d2,crop,yield_amount):
        pass

    def transport(self,d1,d2,crop,yield_amount):
        pass

    
class Greedy_price(Transport_Strategy):
    def __init__(self,districts,crops,prices={}):
        super().__init__(districts,crops)
        # print('Greedy_price')
        self.arrival_prices={}
        self.arrival_prices =  prices
    
    def feasible(self,d1,d2,crop,yield_amount):
        if d1.stock[crop.id] - d1.demand_min_dict[crop.id] >= yield_amount:
            return True
        return False
    
    def transport(self,d1,d2,crop,yield_amount):
        d1.stock[crop.id]-=yield_amount
        d2.stock[crop.id]+=yield_amount
        if d2.is_satisfied(crop) and (crop.id,d2.id) in self.unsatisfied:
            self.unsatisfied.remove((crop.id,d2.id))
        self.cost+=d1.coordinates.distance(d2.coordinates)*crop.transport_cost*yield_amount
        self.logs.append([d1.id,d2.id,crop.id,yield_amount])
    
    def start_transportation(self,districts,crops):
        for crps in self.arrival_prices.keys():
            for (d1_id,d2_id) in self.arrival_prices[crps]:
                d1=districts[d1_id]
                d2=districts[d2_id]
                for crop in crops.values():
                    if crps.lower() == crop.name.lower():
                        if not d1.is_satisfied(crop) and d2.can_give(crop):
                            self.transport(d2,d1,crop,min(d1.requirement(crop),d2.amt_give(crop)))

class Greedy(Transport_Strategy):
    def __init__(self,districts,crops,d=0):
        super().__init__(districts,crops)
        self.distances={}
        if d == 0:
        # print('Greedy')
            for d1 in districts.values():
                for d2 in districts.values():
                    if d1.id != d2.id:
                        self.distances[(d1.id,d2.id)]=d1.coordinates.distance(d2.coordinates)
            self.distances = dict(sorted(self.distances.items(), key=lambda item: item[1]))
        else:
            self.distances = d
 
    def feasible(self,d1,d2,crop,yield_amount):
        if d1.stock[crop.id] - d1.demand_min_dict[crop.id] >= yield_amount:
            return True
        return False

    def transport(self,d1,d2,crop,yield_amount):
        d1.stock[crop.id]-=yield_amount
        d2.stock[crop.id]+=yield_amount
        if d2.is_satisfied(crop) and (crop.id,d2.id) in self.unsatisfied:
            self.unsatisfied.remove((crop.id,d2.id))
        self.cost+=d1.coordinates.distance(d2.coordinates)*crop.transport_cost*yield_amount
        self.logs.append([d1.id,d2.id,crop.id,yield_amount])

    def start_transportation(self,districts,crops):
        for (d1_id,d2_id) in self.distances.keys():
            d1=districts[d1_id]
            d2=districts[d2_id]
            for crop in crops.values():
                if not d1.is_satisfied(crop) and d2.can_give(crop):
                    self.transport(d2,d1,crop,min(d1.requirement(crop),d2.amt_give(crop)))

    
# class Greedy_price(Transport_Strategy):
#     def __init__(self,districts,crops,prices={}):
#         super().__init__(districts,crops)
#         # print('Greedy_price')
#         self.arrival_prices={}
#         self.arrival_prices =  prices
    
#     def feasible(self,d1,d2,crop,yield_amount):
#         if d1.stock[crop.id] - d1.demand_min_dict[crop.id] >= yield_amount:
#             return True
#         return False
    
#     def transport(self,d1,d2,crop,yield_amount):
#         d1.stock[crop.id]-=yield_amount
#         d2.stock[crop.id]+=yield_amount
#         if d2.is_satisfied(crop) and (crop.id,d2.id) in self.unsatisfied:
#             self.unsatisfied.remove((crop.id,d2.id))
#         self.cost+=d1.coordinates.distance(d2.coordinates)*crop.transport_cost*yield_amount
#         self.logs.append([d1.id,d2.id,crop.id,yield_amount])
    
#     def start_transportation(self,districts,crops):
#         for crps in self.arrival_prices.keys():
#             for (d1_id,d2_id) in self.arrival_prices[crps]:
#                 d1=districts[d1_id]
#                 d2=districts[d2_id]
#                 for crop in crops.values():
#                     if crps.lower() == crop.name.lower():
#                         if not d1.is_satisfied(crop) and d2.can_give(crop):
#                             self.transport(d2,d1,crop,min(d1.requirement(crop),d2.amt_give(crop)))

class Bipartite_price(Transport_Strategy):
    def __init__(self,districts,crops,d=0):
        super().__init__(districts,crops)
        # print('Bipartite')
        self.donors = {}
        self.recivers = {}
        for crop in crops.values():
            self.donors[crop] = []
            self.recivers[crop] = []
            for dist in districts.values():
                temp = dist.can_receive(crop)
                if temp == 0:     #0->receiver, 1->donor, 2-> both
                    self.recivers[crop].append(dist.id)
                elif temp == 1:
                    self.donors[crop].append(dist.id)
                    self.recivers[crop].append(dist.id)
                else: 
                    self.donors[crop].append(dist.id)
                    
        # print('donors:')
        # print(self.donors)
        # print()
        # print('receivers:')
        # print(self.recivers)
        self.distances={}
        self.distances = d

    def feasible(self,d1,d2,crop,yield_amount):
        if d1.stock[crop.id] - d1.demand_min_dict[crop.id] >= yield_amount:
            return True
        return False
    
    def transport(self,d1,d2,crop,yield_amount):
        d1.stock[crop.id]-=yield_amount
        d2.stock[crop.id]+=yield_amount
        if d2.is_satisfied(crop) and (crop.id,d2.id) in self.unsatisfied:
            self.unsatisfied.remove((crop.id,d2.id))
        self.cost+=d1.coordinates.distance(d2.coordinates)*crop.transport_cost*yield_amount
        self.logs.append([d1.id,d2.id,crop.id,yield_amount])

    def start_transportation(self,districts,crops):
        for (d1_id,d2_id) in self.distances.keys():
            d1=districts[d1_id]
            d2=districts[d2_id]
            for crop in crops.values(): #d2 -> d1 tansfer
                if d2.id in self.donors[crop] and d1.id in self.recivers[crop]:
                    if d2.can_give(crop) and d1.can_take(crop):
                        temp = min(d1.requirement(crop),d2.amt_give(crop))
                        if temp>0:
                            self.transport(d2,d1,crop,temp)