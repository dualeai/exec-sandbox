"""Saved instances: a running VM's state, sealed and stored in an object store.

Modules: ids, keyring, codec (64 KiB block records) and head (no I/O); models
(data); pack (body objects in local files); store (the store contract, key
layout, probe); s3_store (S3 I/O); commit (head write protocol); manager (save,
load, admin).
"""
