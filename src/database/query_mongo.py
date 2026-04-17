import asyncio
import json
from collections import Counter
from datetime import datetime
from statistics import mean, stdev

import numpy as np  # for percentile calculation
from ml_consumer import MONGODB_URI
from motor.motor_asyncio import AsyncIOMotorClient

mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client["object-detection"]
collection = db["results"]


async def get_latest_documents_and_times(
    limit=1000, output_file="results_summary.json"
):
    cursor = collection.find(sort=[("_id", -1)], limit=limit)
    documents = await cursor.to_list(length=limit)

    records = []
    times = []
    request_ids = []

    # for i, doc in enumerate(documents):
    #     try:
    #         request_id = doc["request_id"]
    #         request_ids.append(request_id)
    #
    #         request_time = float(doc["Timestamp"])
    #         insert_time = doc["Endtime"]
    #         total_time = insert_time - request_time
    #         times.append(total_time)
    #
    #         record = {
    #             "document_index": i + 1,
    #             "request_id": request_id,
    #             "request_time": datetime.fromtimestamp(request_time).isoformat(),
    #             "insert_time": datetime.fromtimestamp(insert_time).isoformat(),
    #             "total_time_seconds": total_time,
    #         }
    #         records.append(record)
    #
    #     except (KeyError, ValueError, TypeError) as e:
    #         records.append({"document_index": i + 1, "error": str(e), "raw_doc": doc})
    for i, doc in enumerate(documents):
        try:
            request_id = doc["request_id"]
            request_ids.append(request_id)

            request_time = float(doc["Timestamp"])
            queue_publish_time = float(doc.get("queue_publish_time", 0))
            consumer_receive_time = float(doc.get("consumer_receive_time", 0))
            db_insert_done_time = float(doc.get("db_insert_done_time", doc["Endtime"]))

            total_time = db_insert_done_time - request_time
            times.append(total_time)

            queue_delay = None
            if queue_publish_time and consumer_receive_time:
                queue_delay = consumer_receive_time - queue_publish_time

            db_write_latency = None
            if consumer_receive_time and db_insert_done_time:
                db_write_latency = db_insert_done_time - consumer_receive_time

            record = {
                "document_index": i + 1,
                "request_id": request_id,
                "request_time": datetime.fromtimestamp(request_time).isoformat(),
                "queue_publish_time": (
                    datetime.fromtimestamp(queue_publish_time).isoformat()
                    if queue_publish_time
                    else None
                ),
                "consumer_receive_time": (
                    datetime.fromtimestamp(consumer_receive_time).isoformat()
                    if consumer_receive_time
                    else None
                ),
                "db_insert_done_time": datetime.fromtimestamp(
                    db_insert_done_time
                ).isoformat(),
                "total_time_seconds": total_time,
                "queue_delay_seconds": queue_delay,
                "db_write_latency_seconds": db_write_latency,
            }
            records.append(record)

        except (KeyError, ValueError, TypeError) as e:
            records.append({"document_index": i + 1, "error": str(e), "raw_doc": doc})
    # summary = {}
    # if times:
    #     p99 = np.percentile(times, 99)
    #     summary = {
    #         "documents_processed": len(times),
    #         "average_time_seconds": mean(times),
    #         "min_time_seconds": min(times),
    #         "max_time_seconds": max(times),
    #         "stddev_time_seconds": stdev(times) if len(times) > 1 else 0,
    #         "p99_time_seconds": p99,
    #     }
    summary = {}
    if times:
        p99 = np.percentile(times, 99)

        queue_delays = [
            r["queue_delay_seconds"]
            for r in records
            if r.get("queue_delay_seconds") is not None
        ]
        db_write_latencies = [
            r["db_write_latency_seconds"]
            for r in records
            if r.get("db_write_latency_seconds") is not None
        ]

        summary = {
            "documents_processed": len(times),
            "average_time_seconds": mean(times),
            "min_time_seconds": min(times),
            "max_time_seconds": max(times),
            "stddev_time_seconds": stdev(times) if len(times) > 1 else 0,
            "p99_time_seconds": p99,
            "average_queue_delay_seconds": mean(queue_delays) if queue_delays else None,
            "p99_queue_delay_seconds": np.percentile(queue_delays, 99)
            if queue_delays
            else None,
            "average_db_write_latency_seconds": mean(db_write_latencies)
            if db_write_latencies
            else None,
            "p99_db_write_latency_seconds": np.percentile(db_write_latencies, 99)
            if db_write_latencies
            else None,
        }

    request_id_counts = Counter(request_ids)
    duplicates = {rid: count for rid, count in request_id_counts.items() if count > 1}

    output = {
        "records": records,
        "summary": summary,
        "duplicates": duplicates,
    }

    # Write to JSON file
    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Processed {len(records)} records.")
    print(f"Summary saved to {output_file}")


if __name__ == "__main__":

    async def main():
        await get_latest_documents_and_times(limit=10000)

    asyncio.run(main())
