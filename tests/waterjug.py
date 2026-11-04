from collections import deque 
CAP_A , CAP_B , GOAL = 3,14, 12

def next_states(st):
    a,b=st
    states = []
    states.append(((CAP_A,b),'fill a completely'))
    states.append(((a,CAP_B),'fill b completely'))
    states.append(((0,b),'empty a completely'))
    states.append(((a,0),'empty b completely'))

    amount = min(a,CAP_B-b)
    states.append(((a-amount,b+amount),'fill b using a'))
    amount = min(CAP_A-a,b)
    states.append(((a+amount,b-amount),'fill a using b'))
    return states


def bfs():
    q = deque([((0,0),[])])
    seen = {(0,0)}
    while q:
        state , path = q.popleft()
        if GOAL in state :
            return path+ [(state , 'GOAL')]
        
        for ns , action in next_states(state) :
            if ns in seen : 
                continue
            seen.add(ns)
            q.append((ns,path + [(state,action)]))

def dfs():
    st = [((0,0),[])]
    seen = {(0,0)}
    while st : 
        state , path = st.pop()
        if GOAL in state : 
            return path+[(state,'GOAL')]

        for ns, action in reversed(next_states(state)):
            if ns in seen : continue
            seen.add((ns))
            st.append((ns,path+[(state,action)]))
    return None


def show(typee , result):
    print(f"{typee} solution")
    for  state, action in result :
        print(f" {state} , {action}")
    return None

show("bfs",bfs())
show("dfs",dfs())