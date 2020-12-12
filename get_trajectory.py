import pandas as pd
import numpy as np

def makeSingle(states, actions, tupSize=4):
  '''
  Modifies a state and action map to be just one tuple
  per row.
  '''
  keys = list(states.keys())
  nLexemes = actions[keys[0]].shape[1]

  statesR, actionsR = {}, {}
  
  for k in keys:
    sessions = states[k].shape[0]
    statesR[k] = np.reshape(states[k], (nLexemes * sessions, tupSize))
    actionsR[k] = np.reshape(actions[k], (nLexemes * sessions, 1))
  return statesR, actionsR



def trajectory_generator(df, included, nTraj=2000):
  '''
  Returns a generator trajctories from df in 
  incremements of nTrajectories.
  '''
  user_list = sorted(df.user_id.unique())
  for i in range(0, len(user_list), nTraj):
    lb = user_list[i]
    try:
      ub= user_list[i+nTraj]
    except:
      ub = user_list[-1]
    dfRelevant = df.loc[(df.user_id >= lb) & (df.user_id < ub)]
    yield get_traj(dfRelevant, included)



def get_traj(df, incl, rewardFn=None):
    '''
    Returns states and actions for lexemes in the form 
    [h_seen, h_correct, difficulty, delta] for each lexeme
    and actions as [sess_seen] for each item.
    Reward function of form r(hSeen, hCorrect, sSeen, sCorrect, delta)
    '''
    states = {}
    actions = {}
    rewards = {}
    idx_to_lex = {}
    lex_to_idx = {}


    i = 0
    for item in incl:
        idx_to_lex[i] = item
        lex_to_idx[item] = i
        i += 1


    df = df.sort_values(by=['user_id', 'timestamp'])

    df_first_lex = df.groupby('lex_user').head(1)
    max_sess = df.groupby('user_id').max().loc[:, 'session']
    min_sess = df.groupby('user_id').min().loc[:, 'session']

    dfclt = df.loc[:, ['lexeme_id', 'difficulty']].drop_duplicates().set_index('lexeme_id').loc[:, 'difficulty']


    itr = max_sess.items()
    itr2 = min_sess.items()


    while True:
        try:
            usr, mx = next(itr)
            usr2, mn = next(itr2)

            if usr != usr2:
                throw("Error")


            sessions =  int(mx - mn)
            
            states[usr] = np.zeros((sessions + 1, len(incl) * 4))
            actions[usr] = np.zeros((sessions + 1, len(incl)))
            rewards[usr] = np.zeros((sessions + 1, len(incl)))
        except:
            break

    #Fill in table with initial counts and actions for each.
    for r in df_first_lex.itertuples(index=False):
        usr, lex = r.user_id, r.lexeme_id
        
        h_seen, h_corr, s_seen = r.history_seen, r.history_correct, r.session_seen

        d = r.difficulty


        c = lex_to_idx[lex]
        c_s = c * 4

        #Fill in all rows with h_seen, h_corr and delta
        #Will update by incrementing them.
        states[usr][:, c_s] = h_seen
        states[usr][:, c_s + 1] = h_corr
        #Fill in difficulty and deltas
        states[usr][:, c_s + 2] = d
        states[usr][:, c_s + 3] = np.arange(len(states[usr]))

        
        

    l_sess = None
    for r in df.itertuples(index=True):
        usr, sess, lex, s_seen, s_corr = r.user_id, r.session, \
            r.lexeme_id, r.session_seen, r.session_correct
        
        m_sess = min_sess[usr]

        
        c = lex_to_idx[lex]
        c_s = c * 4
        row = sess - m_sess
        
        delta = states[usr][row, c_s+3]

        actions[usr][row, c] = s_seen
        if rewardFn is not None:
            rewards[usr][row, c] = rewardFn(r.history_seen, r.history_correct, s_seen, s_corr, delta)

        #Reset delta here so that we can use it for reward
        states[usr][row:, c_s + 3] = np.arange(len(states[usr][row:, c_s+3]))


        try:
            states[usr][row+1:, c_s] += s_seen
            states[usr][row+1:, c_s+1] += s_corr
        except:
            pass
          
  
    return states, actions, rewards, idx_to_lex , lex_to_idx
