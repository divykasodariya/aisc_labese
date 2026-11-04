import heapq
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)
def manhattan(state):
    distance=0
    for i in range(9):
        tile = state.index(i)
        row = tile//3
        column = tile %3

        goalt = GOAL.index(i)
        grow = goalt//3
        gcolumn = goalt%3

        distance+= abs(row-grow)+ abs(gcolumn - column)
    return distance

def neighbours (state):

    moves = [
        [0,1,'right'],
        [0,-1,'left'],
        [1,0,'up'],
        [-1,0,'down'],
    ]

    i = state.index(0)
    row= i//3
    col = i%3
    result =[]
    for dr , dc , action in moves :
        newr = row+dr
        newc = col+dc
        if 0<= newr and newr<3 and 0<= newc and newc<3 :
            newpos=newr*3 + newc
            newstate = list(state)
            newstate[i], newstate[newpos]= newstate[newpos],newstate[i]
            result.append((tuple(newstate),action))

    return result 

def solve(start):
    heap = [(manhattan(start),0,start)]
    g={start:0}
    parent={start:None}
    action={}
    action[start]="START"
    count=0
    while heap :
        f,cnt,curstate = heapq.heappop(heap)
        
        if curstate == GOAL :
            ans=[]
            while curstate :
                
                ans.append((curstate,action[curstate]))
                curstate=parent[curstate]
            return reversed(ans)


        for ns, actionn in neighbours(curstate):   
            newcost = g[curstate]+1
            if ns not in g or g[ns]> newcost:
                count+=1
                g[ns]=newcost
                heapq.heappush(heap,(manhattan(ns),count,ns))
                parent[ns]=curstate
                action[ns]=actionn
            
    return None

def show(state):

    print(state[:3])
    print(state[3:6])
    print(state[6:])

start = (1, 2, 3, 5, 0, 6, 4, 7, 8)

    # Solve puzzle
solution = solve(start)

if solution is None:
    print("No solution found")

else:
    for i, (state, move) in enumerate(solution):

        print("\nStep", i, "|", move)
        show(state)