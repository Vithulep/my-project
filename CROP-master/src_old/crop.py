class Crop():
    def __init__(self,id,name,filename,transport_cost):
        self.id=id
        self.name=name
        self.filename=filename
        self.transport_cost=transport_cost

    def add_district_details(self,cost_dict, yield_dict, price_dict, demand_max_dict,demand_min_dict):
        self.cost_dict=cost_dict
        self.yield_dict=yield_dict
        self.price_dict=price_dict
        self.demand_max_dict=demand_max_dict
        self.demand_min_dict=demand_min_dict
    
    def get_max_demand(self):
        return self.demand_max_dict
    
    def get_min_demand(self):
        return self.demand_min_dict
    
    def give_dist_details(self):
        return [self.cost_dict,self.yield_dict,self.price_dict,self.demand_max_dict,self.demand_min_dict]

