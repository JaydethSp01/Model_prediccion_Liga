from pymongo import MongoClient

class Database:
    _client = None

    @staticmethod
    def get_client(uri="mongodb://localhost:27017/"):
        if Database._client is None:
            Database._client = MongoClient(uri)
        return Database._client

    @staticmethod
    def get_database(db_name="futnexus"):
        client = Database.get_client()
        return client[db_name]
