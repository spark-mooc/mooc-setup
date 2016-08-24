w = [0.4570046605432202,0.03678382345992604,0.21972387691841333,0.18070571642941538,0.5621719210476178,0.1757931390421934,]
theta = 1.0
var('x,y,z')
p1 = implicit_plot3d((w[0] * x**2 + w[1] * y**2 + w[2] * z**2 + w[3] * sqrt(2) * x*y) + (w[4] * sqrt(2) * x*z + w[5] * sqrt(2) * y*z) == theta, (x, 0, 1), (y, 0, 1), (z, 0, 1))
p2 = point3d(negs[0:4999],size=10,color='red')
p3 = point3d(poss,size=10,color='blue')
show(p1+p2+p3)
#show(p1+p3)
