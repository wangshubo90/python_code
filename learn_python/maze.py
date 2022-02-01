from hashlib import new
import queue
from turtle import distance

 

maze = [
    [0,0,0,1,0,0,1],
    [1,1,0,1,0,1,0],
    [1,1,0,0,0,0,1],
    [1,0,0,1,0,1,1],
    [1,0,1,1,1,1,1],
    [1,0,0,0,0,1,1],
    [1,1,1,1,0,0,0]
]


def findShortestPath(maze, entrance=(0,0), exit=(6,6)):
    rowN = len(maze)
    colN = len(maze[0])

    xstart, ystart = entrance
    xdst, ydst = exit
    visited = [[False for i in range(colN)] for j in range(rowN)]
    visited[xstart][ystart] = True

    queue = []
    queue.append((xstart, ystart, 0))

    while len(queue) != 0:
        xcur, ycur, distance = queue.pop(0)
        if (xcur, ycur) == (xdst, ydst):
            return distance

        for i, j in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            x = xcur + i
            y = ycur + j
            if x >=0 and y>=0 and x<rowN and y<colN and not visited[x][y] and maze[x][y]==0:
                visited[x][y] = True
                print((x,y, distance+1))
                queue.append((x, y, distance+1))
    
    return None


new = findShortestPath(maze)

print(new)