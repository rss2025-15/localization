"""
This is an optional submission for the lab5 written portion autograder. 
This is so you can check to make sure that your understanding is 
correct before you get a headstart on the coding. You will not get 
credit for submitting to this autograder. It's just for your testing. 
This pset will be manually graded
"""

import numpy as np

def makeT(x,y,theta):
    return np.array([[np.cos(theta), -np.sin(theta), x], [np.sin(theta), np.cos(theta), y], [0,0,1]])

def get_pose(T):
    ans = np.zeros((3,))
    ans[0] = T[0, 2]
    ans[1]=T[1, 2]
    ans[2] = np.arccos(T[0,0])
    return ans
def answer_to_1i():
    """
    Return your answer to 1i in a python list, [x, y, theta]
    """
    angle1 = np.pi/6
    angle2 = np.pi*11/60

    ans = np.linalg.inv(makeT(0,0,angle1))@makeT(0.2,0.1,11*np.pi/60)
    print(ans)
    return get_pose(ans)

def answer_to_1ii():
    """
    Return your answer to 1ii in a python list, [x, y, theta]
    """
    pose = answer_to_1i()
    x_k1_k = makeT(pose[0], pose[1], pose[2])
    angle1 = np.pi/3
    dx_world_k=makeT(3,4,angle1)@x_k1_k
    print(dx_world_k)
    return get_pose(dx_world_k)
    
    
    # return np.array([0.2, .1, np.pi/60])+np.array([3,4,np.pi/3])


def answer_to_2():
    """
    Return your answers to 2 in a python list for the values z=0,3,5,8,10
    Each value should be a float
    """
    d= 7
    zmax = 10
    epsilon = 0.1
    sigma = 0.5
    def p_short(z):
        if z<=d:
            return 2/d*(1-z/d)
        else:
            return 0
    
    def p_max(z):
        if z>=zmax-epsilon and z<=zmax:
            return 1/epsilon
        else:
            return 0
    
    def p_rand(z):
        if z<=zmax:
            return 1/zmax
        else:
            return 0
    
    def p_hit(z):
        if z<=zmax:
            return 1/(2*np.pi*sigma**2)**0.5*np.exp(-(z-d)**2/(2*sigma**2))
        else:
            return 0
    
    zvals = [0,3,5,8,10]
    ans = [float(0.74*p_hit(z) + 0.07*p_short(z)+0.07*p_max(z) + 0.12*p_rand(z))for z in zvals]
    return ans

print(answer_to_1ii())