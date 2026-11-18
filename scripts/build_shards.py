"""
build_shards.py

packs front-camera images into shard files of 512 images each, so the GPU
job reads one large file per chunk instead of 512 small ones.

each shard is an uncompressed tar file (JPEGs are already compressed)
holding "<frame_id>.jpg" entries. frames.shard_id records which shard each
frame is in.

only frames without a shard_id are packed, into new shards numbered after
the existing ones, so existing shards never change and reruns do nothing.

"""

import os
import tarfile

import psycopg

from config import DATABASE_URL, DATAROOT, SCHEMA_PATH, shard_filename

SHARD_SIZE = 512
SHARD_DIR = "data/shards"


def shard_path(shard_id):
    return os.path.join(SHARD_DIR, shard_filename(shard_id))


def main():
    with open(SCHEMA_PATH) as schema_file:
        schema_sql = schema_file.read()

    os.makedirs(SHARD_DIR, exist_ok=True)

    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(schema_sql)

        next_shard_id = conn.execute(
            "SELECT COALESCE(MAX(shard_id) + 1, 0) FROM frames"
        ).fetchone()[0]
        todo = conn.execute(
            "SELECT frame_id, image_path FROM frames WHERE shard_id IS NULL ORDER BY frame_id"
        ).fetchall()
        print("frames to pack:", len(todo))

        shards_written = 0
        start_index = 0
        while start_index < len(todo):
            group = todo[start_index:start_index + SHARD_SIZE]
            shard_id = next_shard_id + shards_written

            frame_ids = []
            with tarfile.open(shard_path(shard_id), "w") as tar:
                for frame_id, image_path in group:
                    tar.add(os.path.join(DATAROOT, image_path), arcname=f"{frame_id}.jpg")
                    frame_ids.append(frame_id)

            conn.execute(
                "UPDATE frames SET shard_id = %s WHERE frame_id = ANY(%s)",
                (shard_id, frame_ids),
            )

            shards_written += 1
            print("wrote", shard_path(shard_id), "|", len(group), "frames")
            start_index += SHARD_SIZE

    print("shards written:", shards_written)


if __name__ == "__main__":
    main()