from geopy.distance import great_circle

class Coordinate():
    def __init__(self,latitude,longitude):
        self.latitude=latitude
        self.longitude=longitude

    def distance(self,coordinate):
        return great_circle((self.latitude,self.longitude),(coordinate.latitude,coordinate.longitude)).km
