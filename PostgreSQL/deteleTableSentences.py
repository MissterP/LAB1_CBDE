import psycopg2
from config import load_config
from connect import connect

def delete_table(cursor):

    delete_table = '''
        DROP TABLE IF EXISTS sentences
    '''

    try:

        cursor.execute(delete_table)
        print('Table sentences deleted successfully')

    except (psycopg2.DatabaseError, Exception) as error:

        print(f"Error deleting the table sentences: {error}")
        raise # Re-raise the exception


if __name__ == '__main__':
    
    config = load_config()
    if config is None:
        print('Error: Config of the database not found')
        exit()

    database_transaction = connect(config)
    if database_transaction is None:
        print('Error: Database not found')
        exit()
    
    try:

        with database_transaction:

            with database_transaction.cursor() as cursor:

                delete_table(cursor)

    except (psycopg2.DatabaseError, Exception) as error:

        database_transaction.rollback()
        print(f"Error: {error}")

    finally:
        if database_transaction is not None:
            database_transaction.close()
            print('Database connection closed.')

