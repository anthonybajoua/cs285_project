import pandas as pd
import numpy as np

def timestamp_to_session(x):
    result = pd.DataFrame()
    
    timestamps_sorted = np.unique(np.array(x['timestamp']))
    timestamps_sorted.sort()
    
    
    result['timestamp'] = timestamps_sorted
    result['session'] = np.arange(len(timestamps_sorted), dtype=np.uint16)
    return result


def reduce_df(data):
    """
    Reduces dataframes column types to least bits necesarry.
    """
    for c in data.columns:
        if type(df.loc[0, c]) != str:
            data.loc[:, c] = pd.to_numeric(data[c], downcast='unsigned')
            
            
def assign_colstring_to_num(df, col):
    
    id_tbl = pd.DataFrame()
    
    id_tbl.loc[:, col] = df.loc[:, col]
    id_tbl = id_tbl.drop_duplicates()
    id_tbl.loc[:, 'id'] = range(len(id_tbl))
    id_tbl = id_tbl.set_index(col)
    df.loc[:, col] = df.loc[:, col].map(id_tbl.loc[:, 'id'])



def process_original():
	lang_map = {'de' : 0, 'en': 1, 'es': 2, 'fr': 3, 'it': 4, 'pt': 5}

	df = pd.read_csv("data/settles.acl16.learning_traces.13m.csv")


	#Hash strings and map languages to numbers
	assign_colstring_to_num(df, 'user_id')
	assign_colstring_to_num(df, 'lexeme_id')



	df['learning_language'] = df['learning_language'].map(lang_map)
	df['ui_language'] = df['ui_language'].map(lang_map)


	#Save lexemes in different table. Downcast all data
	df['lexeme_string'] = df.lexeme_string.map(lambda x: x[0: x.find('<')])  
	lex_map = df.loc[:, ['lexeme_id', 'lexeme_string']]
	lex_map = lex_map.drop_duplicates()
	lex_map.to_csv("data/lexeme_map.csv", index=False)
	lex_map=None
	#Trim our dataframe
	df = df.drop(["p_recall", "lexeme_string"], axis=1)
	reduce_df(df)

	#Get difficulties for each item and join that.
	i_d = df.groupby('lexeme_id').apply(\
	    lambda x: x['history_correct'].sum() / x['history_seen'].sum())
	df = df.join(i_d.rename("difficulty"), on='lexeme_id')

	#Hash user_lex and ts_user combos, sort by user and ts
	df['lex_user'] = df.loc[:, 'user_id'].astype(str).apply(hash) + df.loc[:, 'lexeme_id'].astype(str).apply(hash)
	df['ts_user'] = df.loc[:, 'user_id'].astype(str).apply(hash) + df.loc[:, 'timestamp'].astype(str).apply(hash)

	df = df.sort_values(by=['user_id', 'timestamp'])

	#Create sessions table and join it with original
	ts_cntr = df.loc[:, ['user_id', 'timestamp']].\
	                 groupby(['user_id']).apply(timestamp_to_session)

	ts_cntr.index = ts_cntr.index.droplevel(1)
	ts_cntr = ts_cntr.reset_index()
	ts_cntr['ts_user'] = ts_cntr.loc[:, 'user_id'].astype(str).apply(hash) \
	                        + ts_cntr.loc[:, 'timestamp'].astype(str).apply(hash)

	df = df.merge(ts_cntr, right_on=['timestamp', 'user_id'], left_on=['timestamp', 'user_id'])

	df = df.drop(['ts_user_x'], axis=1)

	assign_colstring_to_num(df, 'lex_user')
	assign_colstring_to_num(df, 'ts_user_y')

	df.loc[:, 'difficulty'] = df.loc[:, 'difficulty'].astype(np.float32)
	reduce_df(df)

	df.to_csv("data/cleaned.csv")