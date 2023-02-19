import math
import numpy as np

def isBetween(x, a, b):
    #check if x is between the segment composed by point a and point b
    if (a <= x <= b or a >= x >= b):
        return True
    else:
        return False
    

def line_sphere_intersection(point_A, point_B, sphere_center, radius):
    # line has two points(pointA and pointB)
    # sphere_center(x, y, z)
    # radius is the sphere's radius
    a = (point_B[0] - point_A[0])**2 + (point_B[1] - point_A[1])**2 + (point_B[2] - point_A[2])**2

    b = 2* ((point_B[0] - point_A[0])*(point_A[0] - sphere_center[0]) + 
            (point_B[1] - point_A[1])*(point_A[1] - sphere_center[1]) + 
            (point_B[2] - point_A[2])*(point_A[2] - sphere_center[2]))
    c = (point_A[0] - sphere_center[0])**2 + (point_A[1] - sphere_center[1])**2 + (point_A[2] - sphere_center[2])**2 - radius**2

    delta = b**2 - (4*a*c)

    d1 = (-b+np.sqrt(delta))/(2*a)
    d2 = (-b-np.sqrt(delta))/(2*a)
    
    if (delta < 0) :
        print("delta < 0")
        return None
    elif (delta > 0):
        print("delta > 0")
        # one point
        x1 = point_A[0] + d1*(point_B[0] - point_A[0])
        y1 = point_A[1] + d1*(point_B[1] - point_A[1])
        z1 = point_A[2] + d1*(point_B[2] - point_A[2])
        # the other one
        x2 = point_A[0] + d2*(point_B[0] - point_A[0])
        y2 = point_A[1] + d2*(point_B[1] - point_A[1])
        z2 = point_A[2] + d2*(point_B[2] - point_A[2])

        result = []
        if isBetween((x1, y1, z1), point_A, point_B):
            result.append((x1, y1, z1))
        if isBetween((x2, y2, z2), point_A, point_B):
            result.append((x2, y2, z2))
        if len(result) == 0:
            return None
        else:
            return result
    else:
        print("delta = 0")
        d1 = -b / (2*a)
        # one point
        x1 = point_A[0] + d1*(point_B[0] - point_A[0])
        y1 = point_A[1] + d1*(point_B[1] - point_A[1])
        z1 = point_A[2] + d1*(point_B[2] - point_A[2])
        if isBetween((x1, y1, z1), point_A, point_B):
            return (x1, y1, z1)
        else:
            return None

# one point intersection (work)
# sphere = (0, 0, 0)
# redius = 1
# point_A = (1, 1, 0)
# point_B = (1, -1, 0)
# result = line_sphere_intersection(point_A, point_B, sphere, redius)
# print(result)

# two point intersection (work)
# sphere = (0, 0, 0)
# redius = 1
# point_A = (1, 0, 0)
# point_B = (-1, 0, 0)
# result = line_sphere_intersection(point_A, point_B, sphere, redius)
# print(result)

# no intersection (work)
# sphere = (0, 0, 0)
# redius = 1
# point_A = (2, 0, 0)
# point_B = (2, 2, 0)
# result = line_sphere_intersection(point_A, point_B, sphere, redius)
# print(result)

# no intersection (work)
# suppose to be one, but it is not in the segment 
# sphere = (0, 0, 0)
# redius = 1
# point_A = (2, 0, 0)
# point_B = (3, 0, 0)
# result = line_sphere_intersection(point_A, point_B, sphere, redius)
# print(result)

# one point intersection (work)
# suppose to be two, but one is not in the segment
# sphere = (0, 0, 0)
# redius = 1
# point_A = (0, 0, 0)
# point_B = (3, 0, 0)
# result = line_sphere_intersection(point_A, point_B, sphere, redius)
# print(result)