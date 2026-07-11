from modules.database import db_manager
info = db_manager.client.get_collection(db_manager.collection_name)
print(f"Points in collection: {info.points_count}")