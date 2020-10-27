import pandas as pd
import numpy as np

def get_traj(df, incl):
	'''
	Returns states and actions for lexemes in the form 
	[h_seen, h_correct, difficulty, session] for each lexeme
	and actions as [sess_seen] for each item.
	'''
	states = {}
	actions = {}
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
		except:
			break

	#Fill in difficulties
	for k in states.keys():
		for lex, d in dfclt.iteritems():
			c = lex_to_idx[lex]
			c_s = c * 4

			states[k][:, c_s + 2] = d



	for r in df_first_lex.itertuples(index=False):
		sess, usr, lex = r.session, r.user_id, r.lexeme_id
		
		h_seen, h_corr, s_seen = r.history_seen, r.history_correct, r.session_seen


		c = lex_to_idx[lex]
		c_s = c * 4

		
		#Fill in all rows with h_seen, h_corr and difficulty
		#Will update by incrementing them.
		states[usr][:, c_s] = h_seen
		states[usr][:, c_s + 1] = h_corr
		states[usr][:, c_s + 3] = np.arange(len(states[usr]))

		
		actions[usr][0, c] = s_seen
		
		
	add_arr = np.array([0] * len(states[3][0, :]))


	l_sess = None
	for r in df.itertuples(index=True):
		usr, sess, lex, s_seen, s_corr = r.user_id, r.session, \
			r.lexeme_id, r.session_seen, r.session_correct
		
		m_sess, ma_sess = min_sess[usr], max_sess[usr]

		
		c = lex_to_idx[lex]
		c_s = c * 4
		row = sess - m_sess
		
		states[usr][row:, c_s + 3] = np.arange(len(states[usr][row:, c_s+3]))

		actions[usr][row, c] = s_seen

		try:
			states[usr][row+1:, c_s] += s_seen
			states[usr][row+1:, c_s+1] += s_corr
		except:
			pass
		
		
		#Incremement sessions that aren't the min
		if sess != l_sess:
			l_sess = sess
			states[usr][row, :] = states[usr][row, :] + add_arr
		

	return states, actions, idx_to_lex , lex_to_idx
