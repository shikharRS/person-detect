import math
class Point:
    all_points =[]
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def getId(self):
        k=0
        exist = False

        for i in Point.all_points:
            a = self.getDistance(i.x, i.y)
            if a < 10:
                exist = True
                self.all_points[k]=self
                break
            k+=1
        if not exist:
            Point.all_points.append(self)


        return k




    def getDistance(self,x2,y2):
        return math.sqrt((x2 - self.x) ** 2 + (y2 - self.y) ** 2)

