import time
import numpy as np
import pandas as pd
p = pd.DataFrame((np.arange(16).reshape(4,4)), columns = [0,1,2,3] )    
np.random.seed(2)
n_states= 16
action = ['up', 'down', 'left', 'right']     # available actions
epsilon = 0.9   # greedy police
alpha = 0.1     # learning rate
gamma = 0.9    # discount factor
max_episode = 20    # maximum episodes
fresh_time = 0.3    # fresh time for one move

def build_qtable(n_states, action):
    table= pd.DataFrame(np.zeros((n_states, len(action))),columns = action)
    return table


def act(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > epsilon) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(action)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name

def change_env(s, a):
    s_=s
    reward =0
    if a == 'up':
        if s[0]==0:
            s_[0]=0
        else:
            s_[0]= s[0]-1
    elif a== 'down':
        if s[0]==2 and s[1]==3:
            s_ = [3, 3]
            reward = 1
        elif s[0]==3:
            s_[0]=3
        else:
            s_[0]= s[0]+1
    elif a == 'left':
        if s[1]==0:
            s_[1]=0
            
        else:
            s_[1]=s[1]-1
    elif a == 'right':
        if s[0]==3 and s[1]==2:
            s_ = [3,3]
            reward = 5
        elif s[1]==3:
            s_[1]=3
        else:
            s_[1]= s[1]+1
    
    if s_[0]==2 and s_[1]==3:
        reward=-2
    return s_, reward

def update_env(s, episode, step_counter):
    # This is how environment be updated
    l = [['-']*4,['-']*4,['-']*4,['-']*4] 
    l[3][3] = 'T'
    l[2][3] ='X'
    if (s[0] == 3 and s[1]==3) or (s[0]==2 and s[1]==3):
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')

    else:
        l[s[0]][s[1]]='o'
        for i in l:
            for j in i:
                print(j,end='')
            print()
        time.sleep(fresh_time)
        l[s[0]][s[1]]='-'
         
def rl():
    q_table = build_qtable(n_states, action)
    for episode in range(max_episode):
        step_counter = 0
        s = [0, 0]
        
        is_terminated = True
        update_env(s, episode, step_counter)
        while is_terminated:
            S = p.iloc[s[0], s[1]]
            A = act(S, q_table)
            
            print(A)
            s_, R = change_env(s, A) 
            # take action & get next state and reward
            S_ = p.iloc[s_[0], s_[1]]
            print(s_)           
            q_predict = q_table.loc[S, A]
            if (s_[0]==3 and s_[1] ==3):
                q_target = R     # next state is terminal
                is_terminated = False
            elif (s_[0]==2 and s_[1] ==3):
                q_target = R     # next state is terminal
                is_terminated = False
            
                 # next state is not terminal
            else:
                q_target = R + gamma * q_table.iloc[S_, :].max()     # terminate this episode

            q_table.loc[S, A] += alpha * (q_target - q_predict)  # update
            s = s_  # move to next state
            
            update_env(s, episode, step_counter+1)
            step_counter += 1
       
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
        

        

            






























