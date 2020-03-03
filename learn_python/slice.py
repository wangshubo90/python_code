import numpy as np 

a = np.array(range(100)).reshape(10,10)

print("a[slice(3)] = \n{}".format(a[slice(3)]))
print("a[slice(1,10,2)] = \n{}".format(a[slice(1,10,2)]))
print("a[slice(0,None,2)] = \n{}".format(a[slice(0,None,2)]))
print("a[(slice(0,None,2),slice(0,None,2))] = \n{}".format(a[(slice(0,None,2),slice(0,None,2))]))

print(a[(range(0,10,2),2)])
print(a[(slice(0,10,2),2)])

print(a[(range(0,10,2),range(0,10,2))])
print(a[(slice(0,10,2),slice(0,10,2))])

a[(slice(None),[2,1,8])]
a[:,[2,1,8]]

