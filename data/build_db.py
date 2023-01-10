import os
import re
import sqlite3
import pandas as pd

def main():
    con = sqlite3.connect('data/matches.db')
    load_csv_into_db(con, 'data/raw_csv')
    con.commit()
    con.close()
    
def load_csv_into_db(con, dir):

    def _load_csv(file_name):
        if re.search('.*euw.*', file_name):
            reg = 'euw'
        elif re.search('.*na.*', file_name):
            reg = 'na'
        elif re.search('.*kr.*', file_name):
            reg = 'kr'
        elif re.search('.*eune.*', file_name):
            reg = 'eune'
        else:
            raise(Exception)

        df = pd.read_csv(file_name)
        df['region'] = reg
        df.columns = df.columns.str.strip()
        return df

    # Dumping .csv from raw_csv into db
    for file_name in os.listdir(dir):
        # Dump file into table called matches
        print(f'Started file {file_name}')
        _load_csv(os.path.join(dir,file_name)).to_sql('matches', con, if_exists='append', index=False)

    # Count number before remove remakes
    print(con.execute('SELECT COUNT(*) FROM matches').fetchone())
    
    # Deleting remakes
    con.execute(f"DELETE FROM matches WHERE is_remake='1'") # Not tested
    # Count number after remove remakes
    print(con.execute('SELECT COUNT(*) FROM matches').fetchone())
    
    # Deleting duplicated matches (according to match_id)
    con.execute(f"""DELETE FROM matches WHERE ROWID NOT IN (
                   SELECT MIN(ROWID) 
                   FROM matches 
                   GROUP BY match_id
                   )""")
    # Count number after remove duplicates
    print(con.execute('SELECT COUNT(*) FROM matches').fetchone())
        

if __name__ == '__main__':
    main()