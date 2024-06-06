from coordinate import Coordinate
class Warehouse():
    def __init__(self,id,name,address,capacity1,cost,Latitude,Longitude):
        self.id=id
        self.name=name
        self.address=address
        self.capacity1=capacity1
        self.cost= cost
        self.coordinates=Coordinate(Latitude,Longitude)
       
        # self.Forwarded= Forwarded