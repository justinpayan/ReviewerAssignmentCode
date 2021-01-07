import numpy as np

U1 = np.eye(5)
U2 = np.eye(5)
U3 = np.eye(5)
U4 = np.eye(5)
U5 = np.eye(5)
U6 = np.eye(5)
U7 = np.eye(5)

U1[4,0] = -1
U2[4,2] = -1
U3[0,1] = -1
U4[0,2] = -1
U5[3,2] = -1
tmp = U6[:,1].copy()
U6[:,1] = U6[:,3]
U6[:,3] = tmp
tmp = U7[:,2].copy()
U7[:,2] = U7[:,4]
U7[:,4] = tmp

U = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(U1, U2), U3), U4), U5), U6), U7)

def test_s(s):
    return (np.sum(s[0:3]) == 5) and (s[2] + s[3] == 3) and (np.sum(s[[0,2,4]]) == 6) and (not np.any(s < 0))

for i in range(10):
    for j in range(10):
        c = np.array([5,3,6,i,j])
        s = np.dot(U,c)
        if test_s(s):
            print(test_s(s), i, j)
            print(s)