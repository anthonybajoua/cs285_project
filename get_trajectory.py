def get_traj(df, incl):
    #Initialize all our relevant dictionaries
    states, actions, idx_to_lex , lex_to_idx = {}, {}, {}, {}



    i = 0
    for item in incl:
        idx_to_lex[i] = item
        lex_to_idx[item] = i
        i += 1



    df = df.sort_values(by=['user_id', 'timestamp'])

    df_first_lex = df.groupby('lex_user').head(1)
    max_sess = df.groupby('user_id').max().loc[:, 'session']
    min_sess = df.groupby('user_id').min().loc[:, 'session']


    states, actions = {}, {}


    itr = max_sess.items()
    itr2 = min_sess.items()


    while True:
        try:
            usr, mx = next(itr)
            usr2, mn = next(itr2)

            if usr != usr2:
                throw("Error")
            
            usr, sessions = int(usr), int(mx - mn)
            states[usr] = np.zeros((sessions + 1, len(incl) * 3))
            actions[usr] = np.zeros((sessions + 1, len(incl)))
        except:
            break


    for r in df_first_lex.itertuples(index=False):
        sess, usr, lex = r.session, r.user_id, r.lexeme_id
        
        h_seen, h_corr = r.history_seen, r.history_correct
        s_seen = r.session_seen
        
        c = lex_to_idx[lex]
        c_s = c * 3
        
        states[usr][0, c_s] = h_seen
        states[usr][0, c_s+1] = h_corr
        try:
            states[usr][1, c_s+2] = -1  
        except:
            pass
        
        actions[usr][0, c] = s_seen
        
        
    add_arr = np.array([0] * len(states[3][0, :]))

    for i in range(len(add_arr)):
        if i % 3 == 2:
            add_arr[i] = 1
        
    last_usr = None
    l_sess = None
    for r in df.itertuples(index=True):
        usr, sess, lex, s_seen, s_corr = r.user_id, r.session, \
            r.lexeme_id, r.session_seen, r.session_correct
        
        m_sess, ma_sess = min_sess[usr], max_sess[usr]

        
        c = lex_to_idx[lex]
        c_s = c * 3
        row = sess - m_sess
        
        actions[usr][0, c] = s_seen
        
        
        if sess != l_sess:
            l_sess = sess
            states[usr][row, :] = np.copy(states[usr][row - 1, :]) + add_arr   
        
        if sess != ma_sess:
            states[usr][row + 1, c_s] += s_seen
            states[usr][row + 1, c_s +1] += s_corr
            #Set to -1 so when we add 1 to it it goes back to 0
            states[usr][row + 1, c_s + 2] = -1

    return states, actions, idx_to_lex , lex_to_idx
