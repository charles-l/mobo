import apsw
import sqlite_utils


class WrappedCursor(object):
    '''A disguisting hack to make sqlite-utils happy when using an
    apsw.Connection.'''

    def __init__(self, obj):
        self._wrapped_obj = obj
        self._desc = None

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_obj, attr)

    @property
    def description(self):
        return self._desc

    def __iter__(self):
        return self._wrapped_obj.__iter__()

    @property
    def lastrowid(self):
        return self.getconnection().last_insert_rowid()

    @property
    def rowcount(self):
        # make sqlite-utils happy, i guess? ¯\_(ツ)_/¯
        # https://github.com/simonw/sqlite-utils/blob/747be6057d09a4e5d9d726e29d5cf99b10c59dea/sqlite_utils/db.py#L1696
        return 1


class WrappedConnection(apsw.Connection):
    def execute(self, *args):
        c = self.cursor()
        w = WrappedCursor(c)

        def exectrace(cursor, sql, bindings):
            w._desc = cursor.description
            return True

        c.setexectrace(exectrace)

        c.execute(*args)
        return w


def load_db(dbfile):
    db_conn = WrappedConnection(dbfile)
    db_conn.setupdatehook(updatef)
    db = sqlite_utils.Database(db_conn)
    if not db['images'].exists():
        db.create_table(
            'images', {'id': int, 'asset_id': str,
                       'x': int, 'y': int, 'z': int, 'w': int, 'h': int},
            pk='id')
        db['images'].create_index(['z'])
    if not db['assets'].exists():
        db.create_table(
            'assets', {'id': str, 'blob': 'BLOB',
                       'source': str, 'type': str},
            pk='id')
    return db


def updatef(ty, dbname, tablename, rowid):
    print(ty, dbname, tablename, rowid)
